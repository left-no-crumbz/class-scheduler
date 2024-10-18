import random
from datetime import datetime, timedelta
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
        self.set_subject_duration

    def set_subject_duration(self) -> None:
        match self._subject_type:
            case SubjectType.SPECIALIZED_LECTURE:
                self._duration = timedelta(hours=2)
            case SubjectType.SPECIALIZED_LAB:
                self._duration = timedelta(hours=3)
            case SubjectType.GENERAL_EDUCATION:
                self._duration = timedelta(hours=1, minutes=30)

    def get_subject_duration(self) -> timedelta:
        return self._duration

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

    def __str__(self) -> str:
        return f"{self._prefix}"


class Block:
    def __init__(
        self, block_num: str, department: Department, year_level: YearLevel
    ) -> None:
        self._block_num = block_num
        self._department = department
        self._year_level = year_level

    def get_block_num(self) -> str:
        return self._block_num

    def get_year_level(self) -> YearLevel:
        return self._year_level

    # get the subjects of a given year level
    def get_level_subjects(self) -> list[Subject]:
        all_subjects = self._department.get_all_subjects()
        return all_subjects[self._year_level]


class TimeSlot:
    def __init__(self, day: Day, start_time: datetime, end_time: datetime) -> None:
        self._day = day
        self._start_time = start_time
        self._end_time = end_time

    def get_day(self) -> Day:
        return self._day


class Schedule:
    def __init__(self, blocks: list[Block], rooms: list[Room]):
        self._blocks = blocks
        self._rooms = rooms
        self._assignments = []  # List of (Block, Subject, Room, TimeSlot) tuples
        self.generate_random_schedule()

    def generate_random_time(self, subject: Subject) -> tuple[datetime, datetime]:
        # generate a start time. combining today() and min.time() sets the date and time to today midnight
        # it sets the start time at 7AM and generates random minutes spanning 14 hours and where each hour
        # has 12 intervals of 5 minutes
        start_time = datetime.combine(
            datetime.today(), datetime.min.time()
        ) + timedelta(hours=7, minutes=random.randint(0, 14 * 12) * 5)

        end_time = start_time + subject.get_subject_duration()

        # If end time is after 9:00 PM, adjust start time
        if end_time.hour >= 21:
            # Start at 9PM and subtract the subject's duration
            start_time = (
                datetime.combine(datetime.today(), datetime.min.time())
                + timedelta(hours=21)
                - subject.get_subject_duration()
            )

            # adjust the end to be at 9PM
            end_time = datetime.combine(
                datetime.today(), datetime.min.time()
            ) + timedelta(hours=21)
        return start_time, end_time

    def schedule_subject(
        self, block: Block, subject: Subject, day: Day, scheduled_days: set
    ) -> None:
        # Choose a random room
        room = random.choice(self._rooms)

        # generate a random start and end time
        start_time, end_time = self.generate_random_time(subject)

        time_slot = TimeSlot(day, start_time, end_time)
        self._assignments.append((block, subject, room, time_slot))
        scheduled_days.add(day)

    def generate_random_schedule(self):
        # Make sure assignments are always new
        self._assignments = []

        # What gets scheduled on Monday, will also be scheduled to the other day
        day_pairs = (
            (Day.MONDAY, Day.THURSDAY),
            (Day.TUESDAY, Day.FRIDAY),
            (Day.WEDNESDAY, None),
        )

        for block in self._blocks:
            # Start scheduling subjects
            for subject in block.get_level_subjects():
                # ensures that the subject is not assigned on the same day.
                scheduled_days = set()

                for day1, day2 in day_pairs:
                    if day1 not in scheduled_days:
                        # when a subject is assigned on one day, it is also assigned to its pair
                        self.schedule_subject(block, subject, day1, scheduled_days)

                    # Handling the None element in the last day pair
                    if day2 and day2 not in scheduled_days:
                        self.schedule_subject(block, subject, day2, scheduled_days)

    def calculate_fitness(self):
        pass


if __name__ == "__main__":
    sir_uly = Instructor("Sir Uly")
    subject = Subject("ATF", [sir_uly], SubjectType.SPECIALIZED_LECTURE)
    dept = Department("CS", {YearLevel.FIRST: [subject]})
    ROOMS = [
        ["SJH-503", RoomType.LECTURE],
        ["SJH-504", RoomType.LAB],
    ]
    rooms = [Room(room_num, room_type) for room_num, room_type in ROOMS]

    block1 = Block("301", dept, year_level=YearLevel.FIRST)

    sched = Schedule([block1], rooms)
    day_pairs = ((Day.MONDAY, Day.THURSDAY), (Day.TUESDAY, Day.FRIDAY))

    for day1, day2 in day_pairs:
        print(day1.name)
        print(day2.name)
