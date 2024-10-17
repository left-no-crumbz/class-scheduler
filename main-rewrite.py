from datetime import timedelta
from enum import Enum, IntEnum


class RoomType(Enum):
    LECTURE = 1
    LAB = 2


class Day(IntEnum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5


class YearLevel(IntEnum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4


class SubjectType(Enum):
    SPECIALIZED_LECTURE = 1
    SPECIALIZED_LAB = 2
    GENERAL_EDUCATION = 3


class Instructor:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name


class Room:
    def __init__(self, room_num: str, room_type: RoomType) -> None:
        self._room_num = room_num
        self._room_type = room_type

    def get_room(self):
        return self._room_num

    def get_room_type(self) -> RoomType:
        return self._room_type


class Subject:
    def __init__(
        self,
        name: str,
        instructors: list[Instructor],
        subject_type: SubjectType,
    ) -> None:
        self._name = name
        self._instructors = instructors
        self._subject_type = subject_type
        self.get_subject_duration

    def get_subject_duration(self) -> None:
        match self._subject_type:
            case SubjectType.SPECIALIZED_LECTURE:
                self._duration = timedelta(hours=2)
            case SubjectType.SPECIALIZED_LAB:
                self._duration = timedelta(hours=3)
            case SubjectType.GENERAL_EDUCATION:
                self._duration = timedelta(hours=1, minutes=30)

    def get_subject_name(self) -> str:
        return self._name

    def get_subject_instructors(self) -> list[Instructor]:
        return self._instructors


class Department:
    def __init__(
        self, prefix: str, year_level_subjects: dict[YearLevel, list[Subject]]
    ) -> None:
        self._prefix = prefix
        self._year_level_subjects = year_level_subjects

    def get_all_subjects(self) -> dict[YearLevel, list[Subject]]:
        return self._year_level_subjects

    # get the subjects of a given year level
    def get_level_subjects(self, year_level: YearLevel) -> list[Subject]:
        return self._year_level_subjects[year_level]


class Block:
    def __init__(
        self,
        block_num: str,
        department: Department,
    ) -> None:
        pass


if "__main__" == "__name__":
    sir_uly = Instructor("Sir Uly")
    subject = Subject("ATF", [sir_uly], SubjectType.SPECIALIZED_LECTURE)
