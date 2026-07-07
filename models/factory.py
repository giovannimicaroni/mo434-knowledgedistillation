from models.vgg_teacher import VGGTeacher
from models.student import (
    MobileStudentModel,
    NanoStudentModel,
    WideStudentModel,
    DeepStudentModel,
    BottleneckStudentModel,
)
from models.resnet_teacher import ResNetTeacher
from models.convNext_teacher import ConvNeXtTeacher

_TEACHERS = {
    "vgg11": VGGTeacher,
    "ResNet34": ResNetTeacher,
    "convnext": ConvNeXtTeacher
}

# Channel count of each teacher's feature map (and classifier input). The student's
# final 1x1 projection adapter maps its backbone output to this dimension so the same
# student architectures can be distilled against any teacher.
TEACHER_FEATURE_DIMS = {
    "vgg11": 512,
    "ResNet34": 512,
    "convnext": 768,
}

# Comparison set: 5 students spanning a range of depth / width / total params.
_STUDENTS = {
    "mobile_cnn": MobileStudentModel,
    "nano_cnn": NanoStudentModel,
    "wide_cnn": WideStudentModel,
    "deep_cnn": DeepStudentModel,
    "bottleneck_cnn": BottleneckStudentModel,
}


def build_teacher(cfg):
    cls = _TEACHERS[cfg.teacher.architecture]
    return cls(
        pretrained=cfg.teacher.pretrained,
        num_classes=cfg.teacher.num_classes,
        weights_path=cfg.teacher.weights_path,
    )


def build_student(cfg):
    cls = _STUDENTS[cfg.student.architecture]
    out_channels = TEACHER_FEATURE_DIMS[cfg.teacher.architecture]
    return cls(out_channels=out_channels)
