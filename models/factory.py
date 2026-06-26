from models.vgg_teacher import VGGTeacher
from models.student import MobileStudentModel, SlimStudentModel, MediumLargeModel, MediumSlimModel
from models.resnet_teacher import ResNetTeacher
from models.convNext_teacher import ConvNeXtTeacher

_TEACHERS = {
    "vgg11": VGGTeacher,
    "ResNet34": ResNetTeacher,
    "convnext": ConvNeXtTeacher
}

_STUDENTS = {
    "mobile_cnn": MobileStudentModel,
    "slim_student": SlimStudentModel,
    "medium_slim_student": MediumSlimModel,
    "medium_large_student": MediumLargeModel
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
    return cls()
