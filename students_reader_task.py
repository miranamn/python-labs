from typing import Union, List, Dict
from collections import namedtuple
from datetime import date
import os.path
import json

LAB_WORK_SESSION_KEYS = ("presence", "lab_work_n", "lab_work_mark", "date")
STUDENT_KEYS = ("unique_id", "name", "surname", "group", "subgroup", "lab_works_sessions")


class LabWorkSession(namedtuple('LabWorkSession', 'presence, lab_work_number, lab_work_mark, lab_work_date')):
    """
    Информация о лабораторном занятии, которое могло или не могло быть посещено студентом
    """

    def __new__(cls, presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date):
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        if LabWorkSession._validate_session(...):
            ...

        raise ValueError(f"LabWorkSession ::"
                         f"incorrect args :\n"
                         f"presence       : {presence},\n"
                         f"lab_work_number: {lab_work_number},\n"
                         f"lab_work_mark  : {lab_work_mark},\n"
                         f"lab_work_date  : {lab_work_date}")

    @staticmethod
    def _validate_session(presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date) -> bool:
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """

        return True

    def __str__(self) -> str:
        """
            Строковое представление LabWorkSession
            Пример:
            {
                    "presence":      1,
                    "lab_work_n":    4,
                    "lab_work_mark": 3,
                    "date":          "15:12:23"
            }
        """
        ...


class Student:
    __slots__ = ('_unique_id', '_name', '_surname', '_group', '_subgroup', '_lab_work_sessions')

    def __init__(self, unique_id: int, name: str, surname: str, group: int, subgroup: int):
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        if not self._build_student(...):
            raise ValueError(...)

    def _build_student(self, unique_id: int, name: str, surname: str, group: int, subgroup: int) -> bool:
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        return True

    def __str__(self) -> str:
        """
        Строковое представление Student
        Пример:
        {
                "unique_id":          26,
                "name":               "Щукарев",
                "surname":            "Даниил",
                "group":              6408,
                "subgroup":           2,
                "lab_works_sessions": [
                    {
                        "presence":      1,
                        "lab_work_n":    1,
                        "lab_work_mark": 4,
                        "date":          "15:9:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    2,
                        "lab_work_mark": 4,
                        "date":          "15:10:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    3,
                        "lab_work_mark": 4,
                        "date":          "15:11:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    4,
                        "lab_work_mark": 3,
                        "date":          "15:12:23"
                    }]
        }
        """
        ...

    @property
    def unique_id(self) -> int:
        """
        Метод доступа для unique_id
        """
        return self._unique_id

    @property
    def group(self) -> int:
        """
        Метод доступа для номера группы
        """
        ...

    @property
    def subgroup(self) -> int:
        """
        Метод доступа для номера подгруппы
        """
        ...

    @property
    def name(self) -> str:
        """
        Метод доступа для имени студента
        """
        ...

    @property
    def surname(self) -> str:
        """
        Метод доступа для фамилии студента
        """
        ...

    @name.setter
    def name(self, val: str) -> None:
        """
        Метод для изменения значения имени студента
        """
        ...

    @surname.setter
    def surname(self, val: str) -> None:
        """
        Метод для изменения значения фамилии студента
        """
        ...

    @property
    def lab_work_sessions(self):
        """
        Метод доступа для списка лабораторных работ, которые студент посетил или не посетил
        """
        ...

    def append_lab_work_session(self, session: LabWorkSession):
        """
        Метод для регистрации нового лабораторного занятия
        """
        ...


def _load_lab_work_session(json_node) -> LabWorkSession:
    """
        Создание из под-дерева json файла экземпляра класса LabWorkSession.
        hint: чтобы прочитать дату из формата строки, указанного в json используйте
        date(*tuple(map(int, json_node['date'].split(':'))))
    """
    for key in LAB_WORK_SESSION_KEYS:
        if key not in json_node:
            raise KeyError(f"load_lab_work_session:: key \"{key}\" not present in json_node")
    return ...


def _load_student(json_node) -> Student:
    """
        Создание из под-дерева json файла экземпляра класса Student.
        Если в процессе создания LabWorkSession у студента случается ошибка,
        создание самого студента ломаться не должно.
    """
    for key in STUDENT_KEYS:
        ...
    student = ...
    for session in json_node['lab_works_sessions']:
        ...
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
    # csv header
    #     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
    # unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        return None
    ...


def load_students_json(file_path: str) -> Union[List[Student], None]:
    """
    Загрузка списка студентов из json файла.
    Ошибка создания экземпляра класса Student не должна приводить к поломке всего чтения.
    """
    ...


def save_students_json(file_path: str, students: List[Student]):
    """
    Запись списка студентов в json файл
    """
    ...


def save_students_csv(file_path: str, students: List[Student]):
    """
    Запись списка студентов в csv файл
    """
    ...


if __name__ == '__main__':
    # Задание на проверку json читалки:
    # 1. прочитать файл "students.json"
    # 2. сохранить прочитанный файл в "saved_students.json"
    # 3. прочитать файл "saved_students.json"
    # Задание на проверку csv читалки:
    # 1.-3. аналогично

    students = load_students_json('students.json')
    students = save_students_json('students_saved.json')
    load_students_json('students_saved.json', students)

    # students = load_students_csv('students.csv')
    # students = save_students_csv('students_saved.csv')
    # load_students_csv('students_saved.csv', students)

    for s in students:
        print(s)
