import random
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from itertools import combinations

DEBUG = True


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

    def __str__(self) -> str:
        return f"{self._room_num}"


class Subject:
    def __init__(
        self,
        name: str,
        available_instructors: list[Instructor],
        subject_type: SubjectType,
    ) -> None:
        self._name = name
        self._available_instructors = available_instructors
        self._subject_type = subject_type
        self._duration = None
        self._instructor = None
        self.set_subject_duration()
        self.set_subject_instructor()

    def set_subject_instructor(self) -> None:
        self._instructor = random.choice(self._available_instructors)

    def set_subject_duration(self) -> None:
        match self._subject_type:
            case SubjectType.SPECIALIZED_LECTURE:
                self._duration = timedelta(hours=2)
            case SubjectType.SPECIALIZED_LAB:
                self._duration = timedelta(hours=3)
            case SubjectType.GENERAL_EDUCATION:
                self._duration = timedelta(hours=1, minutes=30)

    def get_subject_duration(self) -> timedelta | None:
        return self._duration

    def get_subject_name(self) -> str:
        return self._name

    def get_subject_instructors(self) -> list[Instructor]:
        return self._available_instructors

    def __str__(self) -> str:
        return f"{self._name}"


class Department:
    def __init__(
        self, prefix: str, year_level_subjects: dict[YearLevel, list[Subject]]
    ) -> None:
        self._prefix = prefix
        self._year_level_subjects = year_level_subjects

    def get_prefix(self) -> str:
        return self._prefix

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

    def __str__(self) -> str:
        return f"{self._department.get_prefix()}-{self.get_block_num()}"


class TimeSlot:
    def __init__(self, day: Day, start_time: datetime, end_time: datetime) -> None:
        self._day = day
        self._start_time = start_time
        self._end_time = end_time

    def get_day(self) -> Day:
        return self._day

    def __str__(self):
        return f"{self._start_time.strftime('%I:%M %p')} - {self._end_time.strftime('%I:%M %p')}"


class Schedule:
    def __init__(self, blocks: list[Block], rooms: list[Room]):
        self._blocks = blocks
        self._rooms = rooms
        self._assignments = []  # List of (Block, Subject, Room, TimeSlot) tuples
        self.generate_random_schedule()

    def generate_random_time(self, subject: Subject) -> tuple[datetime, datetime]:
        subject_duration = subject.get_subject_duration()
        if subject_duration is None:
            raise ValueError("Subject duration cannot be None")

        # generate a start time. combining today() and min.time() sets the date and time to today midnight
        start_time = datetime.combine(
            datetime.today(), datetime.min.time()
        ) + timedelta(hours=7, minutes=random.randint(0, 14 * 12) * 5)

        end_time = start_time + subject_duration

        # If end time is after 9:00 PM, adjust start time
        if end_time.hour >= 21:
            # Start at 9PM and subtract the subject's duration
            start_time = (
                datetime.combine(datetime.today(), datetime.min.time())
                + timedelta(hours=21)
                - subject_duration
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

        # For now the length of the assignments is 5 since it assigns a schedule per day
        scheduled_days.add(day)

    def generate_random_schedule(self) -> None:
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

                # This will alternate the assignment of each subject.
                # Ex. subjects = [ATF, ADET, IASEC, IMODSIM]
                # ATF will be assigned on Monday and Thursday
                # ADET will be assigned on Tuesday and Friday
                # IASEC will be assigned on Wednesday only
                for day1, day2 in day_pairs:
                    if day1 not in scheduled_days:
                        # when a subject is assigned on one day, it is also assigned to its pair
                        self.schedule_subject(block, subject, day1, scheduled_days)

                    # Handling the None element in the last day pair
                    if day2 and day2 not in scheduled_days:
                        self.schedule_subject(block, subject, day2, scheduled_days)

    def calculate_fitness(self):
        # initialize the weights of each conflict
        conflicts = defaultdict(int)
        total_assignments = len(self._assignments)

        def time_overlap(t1: TimeSlot, t2: TimeSlot) -> bool:
            # the second condition is the more important part in checking if
            # the times do overlap
            return t1._start_time < t2._end_time and t2._start_time < t1._end_time

        def time_diff_minutes(t1: datetime, t2: datetime) -> int:
            return abs(int((t1 - t2).total_seconds() / 60))

        day_block_assignments = defaultdict(list)
        day_assignments = defaultdict(list)

        # unpack the values in self._assignments
        for block, subject, room, time in self._assignments:
            day = time.get_day()
            """
            day_block_assignments = {
                ('Monday', 'Block1'): [(Math, Room101, Time(9:00-10:00)), (Physics, Room102, Time(9:30-10:30))],
                ('Monday', 'Block2'): [(Chemistry, Room103, Time(11:00-12:00))],
                ('Tuesday', 'Block1'): [(English, Room104, Time(9:00-10:00)), (History, Room105, Time(10:00-11:00))],
            }
            """
            # Compare classes that are on the same day and in the same block,
            # which are the only ones that could potentially conflict.
            day_block_assignments[(day, block)].append((subject, room, time))
            day_assignments[day].append((block, subject, room, time))

        for (day, block), assignments in day_block_assignments.items():
            """
                To visualize sorted_assignments better: 
                --- MONDAY, CS-301 ---
                Subject: ATF in SJH-504 from 2024-10-18 15:00:00 to 2024-10-18 17:00:00
                Instructors: ['Sir Uly']

                --- THURSDAY, CS-301 ---
                Subject: ATF in SJH-503 from 2024-10-18 16:45:00 to 2024-10-18 18:45:00
                Instructors: ['Sir Uly']

                --- TUESDAY, CS-301 ---
                Subject: ATF in SJH-504 from 2024-10-18 19:00:00 to 2024-10-18 21:00:00
                Instructors: ['Sir Uly']
            """
            sorted_assignments = sorted(assignments, key=lambda x: x[2]._start_time)

            for (subject1, room1, time1), (subject2, room2, time2) in combinations(
                sorted_assignments, 2
            ):
                if time_overlap(time1, time2):
                    conflicts["time"] += 1

                    # 0 if False, 1 if True
                    conflicts["room"] += room1 == room2

                    # conflicts["instructor"] += bool(
                    #     set(subject1.get_subject_instructors())
                    #     & set(subject2.get_subject_instructors())
                    # )

                    if subject1._instructor == subject2._instructor:
                        conflicts["instructor"] += 1

                interval = time_diff_minutes(time2._start_time, time1._end_time)

                # Penalize if interval is less than 5 minutes
                if interval < 5:
                    conflicts["interval"] += 1

                # Penalize if interval exceeds 10 and is not divisible by 5
                elif interval > 10 and interval % 5 != 0:
                    conflicts["interval"] += 1

            if DEBUG:
                print(f"\n--- {day.name}, {block} ---")
                for subject, room, time in sorted_assignments:
                    print(
                        f"Subject: {subject.get_subject_name()} in {room} from {time._start_time} to {time._end_time}"
                    )
                    print(
                        f"Instructors: {[instr.get_name() for instr in subject.get_subject_instructors()]}"
                    )

        # For cases where block1 and block2 have overlapping or similar schedule
        for day, assignments in day_assignments.items():
            sorted_assignments = sorted(
                assignments, key=lambda x: x[3]._start_time
            )  # Sort by time

            for (block1, subject1, room1, time1), (
                block2,
                subject2,
                room2,
                time2,
            ) in combinations(sorted_assignments, 2):
                # If block1 and block 2 are different blocks
                # and the time overlaps, penalize the schedule
                if block1 != block2 and time_overlap(time1, time2):
                    # If block1 and block2 have overlapping schedules,
                    # penalize if they have the same room, same instructors
                    if room1 == room2:
                        conflicts["room"] += 1
                        conflicts["time"] += 1

                    # conflicts["instructor"] += bool(
                    #     set(subject1.get_subject_instructors())
                    #     & set(subject2.get_subject_instructors())
                    # )

                    # Check for same subject in different blocks
                    if subject1 == subject2:
                        if subject1._instructor == subject2._instructor:
                            conflicts["instructor"] += 1

        weights = {
            "time": 20,
            "room": 3,
            "instructor": 3,
            "block": 3,
            "interval": 1,
        }

        total_conflicts = sum(conflicts[k] * weights[k] for k in weights)
        max_possible_conflicts = total_assignments * sum(weights.values())
        fitness = 1 - (total_conflicts / max_possible_conflicts)

        if DEBUG:
            for (day, block), assignments in day_block_assignments.items():
                print(f"\n{day.name}, {block}:")
                for subject, room, time in assignments:
                    print(
                        f"  Subject: {subject.get_subject_name()} in {room} from {time._start_time} to {time._end_time}"
                    )
                    print(
                        f"    Instructors: {[instr.get_name() for instr in subject.get_subject_instructors()]}"
                    )
            print(total_conflicts)
            print(max_possible_conflicts)
            print(fitness)


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
    sched.calculate_fitness()
