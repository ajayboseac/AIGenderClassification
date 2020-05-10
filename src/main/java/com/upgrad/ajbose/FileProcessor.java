package com.upgrad.ajbose;

import java.io.*;

public class FileProcessor {

    static void handleNewLines(String fileName) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(fileName)));
        File outputFile = new File( "src/main/resources/output.csv");
        if (outputFile.exists()) {
            outputFile.delete();
        }
        outputFile.createNewFile();
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));

        String line;
        StringBuffer buffer = null;
        while ((line = bufferedReader.readLine()) != null) {
            String[] split = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
            if (split.length >= 26) {
                writer.write(line);
                writer.write("\n");
                buffer=null;
            } else {
                if (buffer == null) {
                    buffer = new StringBuffer();
                    buffer.append(line);
                }else {
                    buffer.append("<br>");
                    buffer.append(line);
                    String[] split1 = buffer.toString().
                            split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                    if(split1.length >= 26){
                        writer.write(buffer.toString());
                        writer.write("\n");
                        buffer =null;
                    }
                }
            }
        }
        writer.flush();
        writer.close();
    }

}
