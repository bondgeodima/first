import os

import sqlite3
from sqlite3 import Error

import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMAGE_DIR_OUT = 'F:/tmp/'
filename_out = 'tmp.jpg'
your_path = os.path.join(IMAGE_DIR_OUT, filename_out)


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks")

    rows = cur.fetchall()

    for row in rows:
        print(row)


def select_task_by_priority(conn, priority):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks WHERE priority=?", (priority,))

    rows = cur.fetchall()

    for row in rows:
        print(row)


def select_10_rows(conn):
    """
    Query 10 rows
    :param conn:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM t LIMIT 10")

    rows = cur.fetchall()

    for row in rows:
        print(row[7])
        im = Image.open(io.BytesIO(row[7]))
        nx, ny = im.size

        # imgplot = plt.imshow(im)
        # plt.show()

        im2 = im.resize((int(nx * 4), int(ny * 4)), Image.BICUBIC)
        im2.save(your_path, dpi=(576, 576))

        img = mpimg.imread(your_path)
        imgplot = plt.imshow(img)
        plt.show()


def main():
    database = r"F:\car_project\1002.773.sqlitedb"

    # create a database connection
    conn = create_connection(database)
    with conn:
        # print("1. Query task by priority:")
        # select_task_by_priority(conn, 1)

        # print("2. Query all tasks")
        # select_all_tasks(conn)

        print("3. Select 10 rows:")
        select_10_rows(conn)


if __name__ == '__main__':
    main()