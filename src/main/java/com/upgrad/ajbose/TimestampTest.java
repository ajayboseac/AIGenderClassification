package com.upgrad.ajbose;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class TimestampTest {
    static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("mm/dd/yyyy hh:mm:ss aa");

    public static void main(String[] args) throws ParseException {
        String timeStamp="7/23/2010  5:08:00 AM";
        Date date = DATE_FORMAT.parse(timeStamp);
        System.out.println(date.toString());
        System.out.println(date.getTime()); ;
    }
}
