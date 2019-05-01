class Student:
    # 初始化学生信息
    def __init__(self, student_id, student_name, student_age):
        self.student_id = student_id
        self.student_name = student_name
        self.student_age = student_age
        self.student_chosen_subject = []
    # 重置学生ID

    def set_id(self, student_id):
        self.studentid = student_id
    # 重置学生姓名

    def set_name(self, name):
        self.name = name
    # 重置学生年龄

    def set_age(self, age):
        self.age = age
    # 选择课程

    def chose_subject(self, subject):
        if subject not in self.student_chosen_subject:
            self.student_chosen_subject.append(subject)
        else:
            print('You have chosen the subject! Please consider another.')

    # 输入学生成绩

    def set_grade(self, subject, grade):
        if subject not in self.student_chosen_subject:
            print('Please confirm the name of the subject.It is not chosen.')
        else:
            subject.set_grade(grade)
    # 输出学生全部的课程成绩

    def get_all_grade(self):
        print(self.student_name)
        for i in self.student_chosen_subject:
            i.get_grade()
    # 输出学生特定科目的成绩

    def get_special_grade(self, subject):
        if subject not in self.student_chosen_subject:
            print("Please confirm the subject! The student didn't chose the subject.")
        else:
            subject.get_grade()


class Subject:
    def __init__(self):
        self.subject_name = ''
        self.subject_id = ''
        self.subject_period = 0
        self.subject_credit = 0
        self.subject_grade = 0
    # 设置subject的基本信息

    def set_information(self, subject_name, subject_id, subject_period, subject_credit):
        self.subject_name = subject_name
        self.subject_id = id
        self.subject_period = subject_period
        self.subject_credit = subject_credit
    # 设置成绩

    def set_grade(self, grade):
        self.subject_grade = grade
    # 打印成绩

    def get_grade(self):
        print("{}:{}".format(self.subject_name, self.subject_grade))


class Teacher:
    def __init__(self):
        self.name = ''

    def set_grade(self, student, subject, grade):
        if isinstance(student, Student):
            student.set_grade(subject, grade)
        else:
            print("Error!")


zhangsan = Student('1120170124', 'zhangsan', 19)
discrete_math = Subject()
discrete_math.set_information('discrete_math', 1, 64, 1.5)
zhangsan.chose_subject(discrete_math)
wang = Teacher()
wang.set_grade(zhangsan, discrete_math, 94)
zhangsan.get_all_grade()


student_name,student_id,student_age = split(input("Please enter the information of students: name,id,age"),",")
student = Student(student_id,student_name,student_age)