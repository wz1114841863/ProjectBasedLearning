import os
import torch
import logging
import uuid
import torchvision.models as models
import torch.distributed as dist
from quant_modules import Quantizer as Q


logger = logging.getLogger(__name__)
quant_args = {}


def set_quantizer(args):
    """全局存储量化配置"""
    global quant_args
    quant_args.update(
        {
            "mode": args.mode,  # 量化模式
            "wbit": args.wbit,  # 权重量化位数
            "abit": args.abit,  # 激活量化位数
            "args": args,  # 原始参数对象
        }
    )


def set_util_logging(filename):
    """设置日志记录"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename),
            logging.StreamHandler(),
        ],  # 同时输出到文件和终端,便于调试和记录
    )


def tag_info(args):
    """返回标签信息"""
    if args.tag == "":
        return ""
    else:
        return "_" + args.tag


def get_ckpt_path(args):
    """获取检查点保存路径"""
    path = "output"
    if torch.distributed.get_rank() == 0:
        # 主进程创建输出目录
        if not os.path.isdir(path):
            os.mkdir(path)
    # 根据模型和数据集创建子目录
    path = os.path.join(path, args.model + "_" + args.dataset)
    if torch.distributed.get_rank() == 0:
        if not os.path.isdir(path):
            os.mkdir(path)

    pathname = args.mode + "_W" + str(args.wbit) + "A" + str(args.abit) + "_"
    # 分布式同步随机路径名
    num = int(uuid.uuid4().hex[0:4], 16)
    num_tensor = torch.tensor(num).cuda()
    dist.broadcast(num_tensor, 0)  # 广播到所有GPU
    pathname += str(num_tensor.item())
    path = os.path.join(path, pathname)
    if torch.distributed.get_rank() == 0:
        if not os.path.isdir(path):
            os.mkdir(path)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    path = os.path.join(path, "gpu_" + str(torch.distributed.get_rank()))
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def get_ckpt_filename(path, epoch):
    return os.path.join(path, "ckpt_" + str(epoch) + ".pth")


def disable_input_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Q):
            module.disable_input_quantization()


def enable_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Q):
            # print("enabling module:", name)
            module.enable_quantization(name)


def disable_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Q):
            # print("disabling module:", name)
            module.disable_quantization(name)


def get_model(args):
    if args.model == "inception_v3":
        return models.inception_v3(aux_logits=False, pretrained=True)
    else:
        return models.__dict__[args.model](pretrained=True)
