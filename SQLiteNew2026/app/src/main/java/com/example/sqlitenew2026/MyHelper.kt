package com.example.sqlitenew2026

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class MyHelper(context: Context): SQLiteOpenHelper(context,"STDB",null,1) {
    override fun onCreate(p0: SQLiteDatabase?) {
       p0?.execSQL("CREATE TABLE STUDENT(_id integer primary key autoincrement,NAME TEXT,CITY TEXT)")

    }

    override fun onUpgrade(
        p0: SQLiteDatabase?,
        p1: Int,
        p2: Int
    ) {

    }

}
