import json
import os.path
from collections import namedtuple
from datetime import date
from typing import Union, List, Dict

LAB_WORK_SESSION_KEYS = ("date", "presence", "lab_work_n", "lab_work_mark")
STUDENT_KEYS = ("unique_id", "name", "surname", "group", "subgroup", "lab_works_sessions")


class LabWorkSession(namedtuple('LabWorkSession', 'lab_work_date, presence, lab_work_number, lab_work_mark')):
    def __new__(cls, lab_work_date: date, presence: bool, lab_work_number: int, lab_work_mark: int):
        if LabWorkSession._validate_args(lab_work_date, presence, lab_work_number, lab_work_mark):
            return super().__new__(cls, lab_work_date, presence, lab_work_number, lab_work_mark)

    @staticmethod
    def _validate_args(lab_work_date: date, presence: bool, lab_work_number: int, lab_work_mark: int) -> bool:
        if not presence and (lab_work_number != -1 or lab_work_mark != -1 or lab_work_date is not None):
            return False
        elif lab_work_number * lab_work_mark < 0:
            return False
        return True

    def __str__(self) -> str:
        return "\t\t{\n" \
               f"\t\t\t\"date\":          \"{self.lab_work_date.year}:{self.lab_work_date.month}:{self.lab_work_date.day}\",\n" \
               f"\t\t\t\"presence\":      {1 if self.presence else 0},\n" \
               f"\t\t\t\"lab_work_n\":    {self.lab_work_number},\n" \
               f"\t\t\t\"lab_work_mark\": {self.lab_work_mark}\n" \
               "\t\t}"


class Student:
    __slots__ = ('_unique_id', '_name', '_surname', '_group', '_subgroup', '_lab_work_sessions')

    def __init__(self, unique_id: int, name: str, surname: str, group: int, subgroup: int):
        if not Student._validate_args(unique_id, name, surname, group, subgroup):
            raise ValueError()
        self._unique_id = unique_id
        self._name = name
        self._surname = surname
        self._group = group
        self._subgroup = subgroup
        self._lab_work_sessions: List[LabWorkSession] = []

    @staticmethod
    def _validate_args(unique_id: int, name: str, surname: str, group: int, subgroup: int) -> bool:
        if len(name) < 2 or not name.isalpha() or len(surname) < 2 or not surname.isalpha():
            return False
        elif not group >= 0 or not subgroup >= 0 or not unique_id >= 0:
            return False
        else:
            return True

    def __str__(self) -> str:
        sep = ',\n'
        return "\t{\n" \
               f"\t\t\"unique_id\": \"{self._unique_id}\",\n" \
               f"\t\t\"name\":      \"{self._name}\",\n" \
               f"\t\t\"surname\":   \"{self._surname}\",\n" \
               f"\t\t\"group\":     {self._group},\n" \
               f"\t\t\"subgroup\":  {self._subgroup},\n" \
               f"\t\t\"lab_works_sessions\":[\n{sep.join(str(v) for v in self._lab_work_sessions)}]\n" \
               "\t}"

    def append_lab_work_session(self, session: LabWorkSession):
        self._lab_work_sessions.append(session)

    @property
    def unique_id(self) -> int:
        return self._unique_id

    @property
    def group(self) -> int:
        return self._group

    @property
    def subgroup(self) -> int:
        return self._subgroup

    @property
    def name(self) -> str:
        return self._name

    @property
    def surname(self) -> str:
        return self._surname

    @name.setter
    def name(self, val: str) -> None:
        assert isinstance(val, str)
        assert len(val) != 0
        self._name = val

    @surname.setter
    def surname(self, val: str) -> None:
        assert isinstance(val, str)
        assert len(val) != 0
        self._surname = val

    @property
    def lab_work_sessions(self):
        for session in self._lab_work_sessions:
            yield session


def _load_lab_work_session(json_node) -> LabWorkSession:
    for key in LAB_WORK_SESSION_KEYS:
        if key not in json_node:
            raise KeyError(f"load_lab_work_sessions:: key \"{key}\" not present in json_node")
    return LabWorkSession(date(*tuple(map(int, json_node['date'].split(':')))),
                          True if int(json_node['presence']) == 1 else False,
                          int(json_node['lab_work_n']),
                          int(json_node['lab_work_mark']))


def _load_student(json_node) -> Student:
    for key in STUDENT_KEYS:
        if key not in json_node:
            raise KeyError(f"KeyError :: {key}")
    student = Student(int(json_node["unique_id"]),
                      json_node["name"],
                      json_node["surname"],
                      int(json_node["group"]),
                      int(json_node["subgroup"]))
    for lw in json_node["lab_works_sessions"]:
        try:
            student.append_lab_work_session(_load_lab_work_session(lw))
        except Exception:
            print("lab_works_session Exception")
            continue
    return student


# csv header
#     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
# unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
UNIQUE_ID = 0
STUD_NAME = 1
STUD_SURNAME = 2
STUD_GROUP = 3
STUD_SUBGROUP = 4
LAB_WORK_DATE = 5
STUD_PRESENCE = 6
LAB_WORK_NUMBER = 7
LAB_WORK_MARK = 8


def load_students_csv(file_path: str) -> Union[List[Student], None]:
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rt', encoding='utf-8') as input_file:
        next(input_file)
        students: Dict[int, Student] = {}
        for line in input_file:
            s_line = line.split(';')
            try:
                student_id = int(s_line[0])
                if student_id not in students:
                    students.update({student_id: Student(int(s_line[UNIQUE_ID]),
                                                         str(s_line[STUD_NAME]).replace('\"', ''),
                                                         str(s_line[STUD_SURNAME]).replace('\"', ''),
                                                         int(s_line[STUD_GROUP]),
                                                         int(s_line[STUD_SUBGROUP]))})

                students[student_id].append_lab_work_session(
                    LabWorkSession(date(*tuple(map(int, s_line[LAB_WORK_DATE].replace('\"', '').split(':')))),
                                   True if int(s_line[STUD_PRESENCE]) == 1 else False,
                                   int(s_line[LAB_WORK_NUMBER]),
                                   int(s_line[LAB_WORK_MARK])))
            except Exception:
                print("Ошибка чтения")
                continue
    return list(students.values())


def load_students_json(file_path: str) -> Union[List[Student], None]:
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        json_node = json.load(f)
        if 'students' not in json_node:
            return None
        students = []
        for node in json_node['students']:
            try:
                students.append(_load_student(node))
            except Exception:
                print("Ошибка загрузки")
                continue
    return students


def save_students_json(file_path: str, students: List[Student]):
    with open(file_path, 'wt', encoding='utf-8') as output_file:
        sep = ',\n'
        print(f"{{\n\t\"students\":[\n{sep.join(str(v) for v in students)}]\n}}", file=output_file)


def save_students_csv(file_path: str, students: List[Student]):
    with open(file_path, 'wt', encoding='utf-8') as output_file:
        print("unique_id;name;surname;group;subgroup;date;presence;lab_work_number;lab_work_mark", file=output_file)
        for student in students:
            for lab_work in student.lab_work_sessions:
                print(f"{student.unique_id};"
                      f"\"{student.name}\";"
                      f"\"{student.surname}\";"
                      f"{student.group};"
                      f"{student.subgroup};"
                      f"\"{lab_work.lab_work_date.year}:{lab_work.lab_work_date.month}:{lab_work.lab_work_date.day}\";"
                      f"{1 if lab_work.presence else 0};{lab_work.lab_work_number};{lab_work.lab_work_mark}",
                      file=output_file)


if __name__ == '__main__':
    print("JSON")
    #students = load_students_json('students.json')
    # for s in students:
    # print(s)
    #save_students_json('students_saved.json', students)
    #students1 = load_students_json('students_saved.json')
    # for s in students1:
    # print(s)
    print("__________________________________________")
    print("CSV")
    students2 = load_students_csv('students.csv')
    #for s in students2:
        #print(s)
    save_students_csv('students_saved.csv', students2)
    students3 = load_students_csv('students_saved.csv')
    for s in students3:
        print(s)
