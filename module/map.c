#include "map.h"
#include "list.h"
#include "ctrl.h"
#include <linux/version.h>
#include <linux/sched.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/slab.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>

#ifdef _CUDA
#include <nv-p2p.h>

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/atomic.h>
struct gpu_region {
    nvidia_p2p_page_table_t *pages;
    nvidia_p2p_dma_mapping_t *mapping;
    struct pci_dev *peer;
    bool pages_owned, mapping_owned;
    bool module_owned;
    u64 reserved_bytes;
};
static char *dma_peer_bdf;
module_param(dma_peer_bdf, charp, 0444);
MODULE_PARM_DESC(dma_peer_bdf, "Map GPU memory for this DMA peer (DSA PCI BDF), not the virtual NVMe controller");
static unsigned int max_gpu_pinned_mb = 32;
module_param(max_gpu_pinned_mb, uint, 0444);
static atomic64_t gpu_pinned = ATOMIC64_INIT(0);
static atomic_t gpu_cleanup_fault = ATOMIC_INIT(0);
#endif


#define GPU_PAGE_SHIFT  16
#define GPU_PAGE_SIZE   (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK   ~(GPU_PAGE_SIZE - 1)

uint32_t max_num_ctrls = 64;


static struct map* create_descriptor(const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    unsigned long i;
    struct map* map = NULL;

    map = kvmalloc(sizeof(struct map) + (n_pages) * sizeof(uint64_t), GFP_KERNEL);
    if (map == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&map->list);

    map->owner = current;
    map->vaddr = vaddr;
    map->pdev = ctrl->pdev;
    map->page_size = 0;
    map->data = NULL;
    map->release = NULL;
    map->n_addrs = n_pages;
    map->n_dma_addrs = 0;


    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = 0;
    }

    return map;
}



void unmap_and_release(struct map* map)
{
    list_remove(&map->list);

    if (map->release != NULL && map->data != NULL)
    {
        map->release(map);
    }

    /* A failed GPU cleanup retains all owned handles and its module pin. */
    if (map->data) return;
    kvfree(map);
}



struct map* map_find(const struct list* list, u64 vaddr)
{
    const struct list_node* element = list_next(&list->head);
    struct map* map = NULL;

    while (element != NULL)
    {
        map = container_of(element, struct map, list);

        if (map->owner == current)
        {
            if (map->vaddr == (vaddr & PAGE_MASK) || map->vaddr == (vaddr & GPU_PAGE_MASK))
            {
                return map;
            }
        }

        element = list_next(element);
    }

    return NULL;
}



static void release_user_pages(struct map* map)
{
    unsigned long i;
    struct page** pages;
    struct device* dev;

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_dma_addrs; ++i)
    {
        dma_unmap_page(dev, map->addrs[i], PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    pages = (struct page**) map->data;
    for (i = 0; i < map->n_addrs; ++i)
    {
        put_page(pages[i]);
    }

    kvfree(map->data);
    map->data = NULL;

    //printk(KERN_DEBUG "Released %lu host pages\n", map->n_addrs);
}



static long map_user_pages(struct map* map)
{
    unsigned long i;
    long retval;
    struct page** pages;
    struct device* dev;

    pages = (struct page**) kvcalloc(map->n_addrs, sizeof(struct page*), GFP_KERNEL);
    if (pages == NULL)
    {
        printk(KERN_CRIT "Failed to allocate page array\n");
        return -ENOMEM;
    }

#if LINUX_VERSION_CODE <= KERNEL_VERSION(4, 5, 7)
#warning "Building for older kernel, not properly tested"
    retval = get_user_pages(current, current->mm, map->vaddr, map->n_addrs, 1, 0, pages, NULL);
#elif LINUX_VERSION_CODE <= KERNEL_VERSION(4, 8, 17)
#warning "Building for older kernel, not properly tested"
    retval = get_user_pages(map->vaddr, map->n_addrs, 1, 0, pages, NULL);
#elif LINUX_VERSION_CODE < KERNEL_VERSION(6, 5, 0)
    /* Linux 6.5 起 get_user_pages 移除了最后的 vmas 参数。 */
    retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages, NULL);
#else
    retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages);
#endif
    if (retval <= 0)
    {
        kvfree(pages);
        printk(KERN_ERR "get_user_pages() failed: %ld\n", retval);
        return retval < 0 ? retval : -ENOMEM;
    }

    map->page_size = PAGE_SIZE;
    map->data = (void*) pages;
    map->release = release_user_pages;
    if (map->n_addrs != retval) {
        map->n_addrs = retval;
        return -ENOMEM;
    }

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = dma_map_page(dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);

        retval = dma_mapping_error(dev, map->addrs[i]);
        if (retval != 0)
        {
            printk(KERN_ERR "Failed to map page for some reason\n");
            return -EIO;
        }
        map->n_dma_addrs++;
       // printk("map_user_page: device: %02x:%02x.%1x\tvaddr: %llx\ti: %lu\tdma_addr: %llx\n", map->pdev->bus->number, PCI_SLOT(map->pdev->devfn), PCI_FUNC(map->pdev->devfn), (uint64_t) map->vaddr, i, map->addrs[i]);
    }

    return 0;
}



struct map* map_userspace(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    long err;
    struct map* md;

    if (n_pages < 1 || n_pages > (SIZE_MAX - sizeof(struct map)) / sizeof(uint64_t) || n_pages > SIZE_MAX / GPU_PAGE_SIZE)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = PAGE_SIZE;

    err = map_user_pages(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu host pages starting at address %llx\n", 
    //        md->n_addrs, md->vaddr);
    return md;
}



#ifdef _CUDA
static void release_gpu_memory(struct map *map)
{
    struct gpu_region *gd = map->data;
    int ret;
    if (!gd) return;
    if (gd->mapping_owned) {
        ret = nvidia_p2p_dma_unmap_pages(gd->peer, gd->pages, gd->mapping);
        if (ret) goto quarantine;
        gd->mapping_owned = false;
    }
    if (gd->pages_owned) {
        ret = nvidia_p2p_put_pages_persistent(map->vaddr, gd->pages, 0);
        if (ret) goto quarantine;
        gd->pages_owned = false;
    }
    atomic64_sub(gd->reserved_bytes, &gpu_pinned);
    if (gd->peer) pci_dev_put(gd->peer);
    {
        bool owned = gd->module_owned;
        kfree(gd); map->data = NULL;
        if (owned) module_put(THIS_MODULE);
    }
    return;
quarantine:
    atomic_set(&gpu_cleanup_fault, 1);
    pr_err("libnvm: GPU cleanup failed: %d; retaining mappings and module, refusing new GPU maps\n", ret);
}
static int map_gpu_memory(struct map *map, struct list *list)
{
    struct gpu_region *gd;
    nvidia_p2p_page_table_t *pages = NULL;
    nvidia_p2p_dma_mapping_t *mapping = NULL;
    unsigned int domain, bus, slot, fn;
    unsigned long i;
    char extra;
    int ret;
    u64 bytes = GPU_PAGE_SIZE * map->n_addrs;
    if (atomic_read(&gpu_cleanup_fault)) return -EIO;
    if (!max_gpu_pinned_mb || bytes > (u64)max_gpu_pinned_mb * 1024 * 1024) return -ENOSPC;
    if (atomic64_add_return(bytes, &gpu_pinned) > (u64)max_gpu_pinned_mb * 1024 * 1024) {
        atomic64_sub(bytes, &gpu_pinned); return -ENOSPC;
    }
    gd = kzalloc(sizeof(*gd), GFP_KERNEL);
    if (!gd) { atomic64_sub(bytes, &gpu_pinned); return -ENOMEM; }
    gd->reserved_bytes = bytes;
    map->data = gd; map->release = release_gpu_memory;
    if (dma_peer_bdf && *dma_peer_bdf) {
        if (sscanf(dma_peer_bdf, "%x:%x:%x.%x%c", &domain, &bus, &slot, &fn, &extra) != 4 ||
            domain > 65535 || bus > 255 || slot > 31 || fn > 7) return -EINVAL;
        gd->peer = pci_get_domain_bus_and_slot(domain, bus, PCI_DEVFN(slot, fn));
    } else gd->peer = pci_dev_get(map->pdev);
    if (!gd->peer) return -ENODEV;
    /* Keep the helper resident until this persistent mapping is released.
     * An abandoned process therefore cannot trigger unsafe module-exit cleanup. */
    if (!try_module_get(THIS_MODULE)) return -EBUSY;
    gd->module_owned = true;
    /* Failed API outputs are not owned handles. */
    ret = nvidia_p2p_get_pages_persistent(map->vaddr, bytes, &pages, 0);
    if (ret) {
        pr_err("libnvm: persistent GPU pin failed: %d, va=%#llx bytes=%llu peer=%s\n", ret,
               map->vaddr, bytes, pci_name(gd->peer));
        return ret;
    }
    if (!pages) return -EIO;
    gd->pages = pages; gd->pages_owned = true;
    if (!NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE(pages) ||
        pages->page_size != NVIDIA_P2P_PAGE_SIZE_64KB || pages->entries < map->n_addrs) return -EOPNOTSUPP;
    ret = nvidia_p2p_dma_map_pages(gd->peer, pages, &mapping);
    if (ret) {
        pr_err("libnvm: GPU DMA map failed: %d, peer=%s\n", ret, pci_name(gd->peer));
        return ret;
    }
    if (!mapping) return -EIO;
    gd->mapping = mapping; gd->mapping_owned = true;
    if (!NVIDIA_P2P_DMA_MAPPING_VERSION_COMPATIBLE(mapping) ||
        mapping->page_size_type != NVIDIA_P2P_PAGE_SIZE_64KB ||
        mapping->entries < map->n_addrs || !mapping->dma_addresses) return -EOPNOTSUPP;
    for (i = 0; i < map->n_addrs; ++i) map->addrs[i] = mapping->dma_addresses[i];
    return 0;
}
#endif

#ifdef _CUDA
struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages, struct list* ctrl_list)
{
    int err;
    struct map* md = NULL;

    if (n_pages < 1 || n_pages > (SIZE_MAX - sizeof(struct map)) / sizeof(uint64_t) || n_pages > SIZE_MAX / GPU_PAGE_SIZE)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & GPU_PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = GPU_PAGE_SIZE;
    md->ctrl_list = ctrl_list;
    err = map_gpu_memory(md, ctrl_list);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu GPU pages starting at address %llx\n", 
    //        md->n_addrs, md->vaddr);
    return md;
}
#endif

