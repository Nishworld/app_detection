import sqlite3
sqlite3.connect("restraurant.db")
q1 = """ create table menu (
         id int primary key , 
         name varchar(50) not FULL 
         price number(50)
         """
q2 = """ create table   ""