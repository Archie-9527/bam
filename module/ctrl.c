#include "ctrl.h"
#include "list.h"
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <asm/errno.h>



struct ctrl* ctrl_get(struct list* list, struct class* cls, struct pci_dev* pdev, int number)
{
    struct ctrl* ctrl = NULL;

    ctrl = kmalloc(sizeof(struct ctrl), GFP_KERNEL | GFP_NOWAIT);
    if (ctrl == NULL)
    {
        printk(KERN_CRIT "Failed to allocate controller reference\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&ctrl->list);

    ctrl->pdev = pdev;
    ctrl->number = number;
    ctrl->rdev = 0;
    ctrl->cls = cls;
    ctrl->chrdev = NULL;

    snprintf(ctrl->name, sizeof(ctrl->name), "%s%d", KBUILD_MODNAME, ctrl->number);
    ctrl->name[sizeof(ctrl->name) - 1] = '\0';

    list_insert(list, &ctrl->list);

    return ctrl;
}



void ctrl_put(struct ctrl* ctrl)
{
    if (ctrl != NULL)
    {
        list_remove(&ctrl->list);
        ctrl_chrdev_remove(ctrl);
        kfree(ctrl);
    }
}



struct ctrl* ctrl_find_by_pci_dev(const struct list* list, const struct pci_dev* pdev)
{
    const struct list_node* element = list_next(&list->head);
    struct ctrl* ctrl;

    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        if (ctrl->pdev == pdev)
        {
            return ctrl;
        }

        element = list_next(element);
    }

    return NULL;
}



struct ctrl* ctrl_find_by_inode(const struct list* list, const struct inode* inode)
{
    const struct list_node* element = list_next(&list->head);
    struct ctrl* ctrl;

    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        /* miscdevice 共用分派入口，通过实际设备号区分控制器，不能再比较 inode->i_cdev。 */
        if (ctrl->rdev != 0 && ctrl->rdev == inode->i_rdev)
        {
            return ctrl;
        }

        element = list_next(element);
    }

    return NULL;
}



int ctrl_chrdev_create(struct ctrl* ctrl, dev_t first, const struct file_operations* fops)
{
    int err;
    (void)first;
    if (ctrl->chrdev != NULL)
        return 0;

    /* 使用非 GPL-only 的 miscdevice 接口，兼容 NVIDIA 专有驱动依赖。 */
    memset(&ctrl->miscdev, 0, sizeof(ctrl->miscdev));
    ctrl->miscdev.minor = MISC_DYNAMIC_MINOR;
    ctrl->miscdev.name = ctrl->name;
    ctrl->miscdev.fops = fops;
    ctrl->miscdev.parent = &ctrl->pdev->dev;
    ctrl->miscdev.mode = 0600;
    err = misc_register(&ctrl->miscdev);
    if (err)
        return err;

    ctrl->chrdev = ctrl->miscdev.this_device;
    ctrl->rdev = ctrl->chrdev->devt;
    printk(KERN_INFO "BaM miscdevice /dev/%s ready (%d:%d)\n",
           ctrl->name, MAJOR(ctrl->rdev), MINOR(ctrl->rdev));
    return 0;
}



void ctrl_chrdev_remove(struct ctrl* ctrl)
{
    if (ctrl->chrdev != NULL) {
        misc_deregister(&ctrl->miscdev);
        ctrl->chrdev = NULL;
        ctrl->rdev = 0;
    }
}

