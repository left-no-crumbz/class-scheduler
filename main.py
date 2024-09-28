# Class Scheduler
# -- The goal is to use a genetic algorithm to create a class schedule that has no conflicts

# CONSTRAINTS
# These are the constraints to prevent conflicts from schedules
#  - A professor cannot be in two rooms at once in a given time slot
#  - A room can only accomodate one block per time slot.
#  - A block can only take one subject per time slot
#  - Each subject has their own length in hours which is the time they would occupy a room
#  - The available time is only 7am to 9pm.
#  - A subject can occupy any time slot as long as it doesn't conflict with another subject,
#    it fulfills the subject's length in hours, and is within 7am - 9pm

# Additional Constraints
#  - The available days in a week is Monday to Friday
#  - A subject may only be plotted twice in a single week but not in the same day
#  - After each subject, there should be an interval of between 5-10 minutes
#  - After every 5 hours, there should be a break lasting a range of 30 minutes to 1 hour.
#  - A room must be defined by its type (e.g, Lab, Lecture)
#  - A subject must define what type of room it will use (e.g, Lab, Lecture or both)
#  - A subject should be able to occupy decimal hours (e.g., 1 hour and 30 minutes, etc)
#  - A subject can have multiple available instructors and it randomly selects from the list of instructors.
#  - If a subject has both Lecture and Lab components, it should use a Lab room in the first
#    meeting in a week, and Lecture room in the second week or vice versa.


from datetime import timedelta
from enum import Enum, IntEnum

import numpy as np


class RoomType(Enum):
    LECTURE = 1
    LAB = 2


# TODO: Add a subject type that is Specialized, General Ed, etc.
# pure lecture: 1 hr 30 mins
# lab: 3 hours
# lecture ng lab: 2 hours
# I guess we can do:
# If subjectType == BOTH then it could appear only once
class SubjectType(Enum):
    LECTURE = 1
    LAB = 2
    BOTH = 3


class YearLevel(IntEnum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4


class Instructors:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return f"Instructor: {self.name}"


class Room:
    def __init__(self, room_num: str, max_capacity: int, room_type: RoomType) -> None:
        self.room_num = room_num
        self.max_capacity = max_capacity
        self.room_type = room_type

    def get_room_num(self) -> str:
        return self.room_num

    def get_max_capacity(self) -> int:
        return self.max_capacity

    def get_room_type(self) -> RoomType:
        return self.room_type

    def get_room_type_str(self) -> str:
        return self.room_type.name

    def __str__(self) -> str:
        return f"Room Number: {self.room_num}, Max Capacity: {self.max_capacity}, Room Type: {self.room_type.name}"


class Subject:
    def __init__(
        self,
        name: str,
        instructors: list[Instructors],
        subject_type: SubjectType,
        duration: timedelta,
    ) -> None:
        self.name = name
        self.instructors = instructors
        self.subject_type = subject_type
        self.duration = duration

    def get_subject_name(self) -> str:
        return self.name

    def get_subject_type(self) -> SubjectType:
        return self.subject_type

    def get_instructors(self) -> list[Instructors]:
        return self.instructors

    def get_subject_duration(self) -> timedelta:
        return self.duration

    def __str__(self) -> str:
        instructors_np = np.array(self.instructors)
        to_str = np.vectorize(lambda instructor: str(instructor))
        to_str(instructors_np)

        return f"Subject: {self.name}, Type: {self.subject_type.name}, Instructors: [{", ".join(to_str(instructors_np))}], Duration: {self.duration}"


class Department:
    def __init__(self, prefix: str, subjects: dict[YearLevel, list[Subject]]) -> None:
        self.prefix = prefix  # CS, WD, NA, EMC, CYB

        # Offered subjects by each department depending on their yearlevel
        # i.e., DASALGO, ADMATHS for CS
        # ANAGEOM for WD, etc
        self.subjects = subjects

    def get_dept_prefix(self) -> str:
        return self.prefix

    def get_dept_subjects(self) -> dict[YearLevel, list[Subject]]:
        return self.subjects

    def __str__(self) -> str:
        return (
            f"Prefix: {self.prefix}, Offered Subjects Per Year Level: {self.subjects}"
        )


if __name__ == "__main__":
    maam_mags = Instructors("Ma'am Avigail Magbag")
    maam_jehan = Instructors("Ma'am Jehan")
    maam_caro = Instructors("Ma'am Caro")

    # TODO: Attach the duration to the SubjectType
    LOGPROG_LECTURE = Subject(
        "LOGPROG", [maam_mags], SubjectType.LECTURE, timedelta(hours=3)
    )
    LOGPROG_LAB = Subject(
        "LOGPROG", [maam_mags], SubjectType.LAB, timedelta(hours=1, minutes=30)
    )
    PHCI = Subject(
        "PHCI", [maam_jehan], SubjectType.LECTURE, timedelta(hours=1, minutes=30)
    )
    OOP_LECTURE = Subject(
        "OOP", [maam_caro], SubjectType.BOTH, timedelta(hours=1, minutes=30)
    )
    OOP_LAB = Subject("OOP", [maam_caro], SubjectType.BOTH, timedelta(hours=3))
    dept = Department(
        "CS",
        {
            YearLevel.FIRST: [LOGPROG_LECTURE, LOGPROG_LAB, PHCI],
            YearLevel.SECOND: [OOP_LAB, OOP_LECTURE],
        },
    )

    print(str(OOP_LECTURE))
