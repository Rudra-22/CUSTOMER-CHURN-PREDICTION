package com.example.sqlitenew2026

import android.content.ContentValues
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.os.Bundle
import android.widget.SimpleCursorAdapter
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.sqlitenew2026.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    lateinit var binding: ActivityMainBinding
    lateinit var db: SQLiteDatabase
    lateinit var rs: Cursor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        var helper = MyHelper(applicationContext)
        db = helper.readableDatabase
        rs = db.rawQuery("SELECT * FROM STUDENT", null)

        loaddata()

        binding.btninsert.setOnClickListener {
            var bin = rs.count
            var cv = ContentValues()
            cv.put("NAME",binding.edname.text.toString())
            cv.put("CITY",binding.edcity.text.toString())
            db.insert("STUDENT",null,cv)
            rs.requery()
            var ain = rs.count
            if(ain > bin){
                Toast.makeText(this,"Inserted",Toast.LENGTH_SHORT).show()
                loaddata()
            }

        }
        binding.btnupdate.setOnClickListener {
            if(binding.edname.text.toString()=="")
            {
                Toast.makeText(this,"Select Record to Update",Toast.LENGTH_SHORT).show()

            }
            else{
                var cv = ContentValues()
                cv.put("NAME",binding.edname.text.toString())
                cv.put("CITY",binding.edcity.text.toString())
                db.update("STUDENT",cv,"_id=?",arrayOf(rs.getString(0)))
                Toast.makeText(this,"Updated",Toast.LENGTH_SHORT).show()
                rs.requery()
                loaddata()
            }

        }
        binding.btndelete.setOnClickListener {
            if(binding.edname.text.toString()==""){
                Toast.makeText(this,"Select Data from List to delete",Toast.LENGTH_SHORT).show()
            }
            else{
                db.delete("STUDENT","_id=?",arrayOf(rs.getString(0)))
                rs.requery()
                loaddata()
            }

        }
        binding.btnclear.setOnClickListener {
            binding.edname.setText("")
            binding.edcity.setText("")
        }
        binding.btnfirst.setOnClickListener {
            if(rs.moveToFirst()){
                binding.edname.setText(rs.getString(1))
                binding.edcity.setText(rs.getString(2))
            }
        }
        binding.btnlast.setOnClickListener {
            if(rs.moveToLast()){
                binding.edname.setText(rs.getString(1))
                binding.edcity.setText(rs.getString(2))
            }
        }
        binding.btnnext.setOnClickListener {
            if(rs.moveToNext()){
                binding.edname.setText(rs.getString(1))
                binding.edcity.setText(rs.getString(2))
            }
            else
            {
                rs.moveToFirst()
                binding.edname.setText(rs.getString(1))
                binding.edcity.setText(rs.getString(2))
            }
        }
        binding.btnprevious.setOnClickListener {
            if(rs.moveToPrevious()){
                binding.edname.setText(rs.getString(1))
                binding.edcity.setText(rs.getString(2))
            }
            else
            {
                rs.moveToLast()
                binding.edname.setText(rs.getString(1))
                binding.edcity.setText(rs.getString(2))
            }
        }



        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

    private fun loaddata() {
        if(rs.count>0){
            val adapter = SimpleCursorAdapter(applicationContext,android.R.layout.simple_list_item_2,rs,
                arrayOf("NAME","CITY"),intArrayOf(android.R.id.text1,android.R.id.text2),0)
            binding.listview.adapter = adapter

            binding.listview.setOnItemClickListener { parent, view, position, id ->
            val cursor = parent.adapter.getItem(position) as Cursor
                val name =cursor.getString(cursor.getColumnIndexOrThrow("NAME"))
                val city =cursor.getString(cursor.getColumnIndexOrThrow("CITY"))
                binding.edname.setText(name)
                binding.edcity.setText(city)




            }

        }
        else{

        }



    }
}