from models.vgg_teacher import VGGTeacher
from models.student import MobileStudentModel
from models.resnet_teacher import ResNetTeacher

_TEACHERS = {
    "vgg11": VGGTeacher,
    "ResNet34": ResNetTeacher,
}

_STUDENTS = {
    "mobile_cnn": MobileStudentModel,
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
