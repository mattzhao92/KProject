import java.io.BufferedReader;
import java.io.*;


public class ValidationFileMaker {

	public static int read_libsvm_stream(SparseData data, int docs, BufferedReader input_file) throws Exception {
		String l;
		String[] splits, s;
		int w= 0;
		
		int test_doc_num = 500000;
		File file = new File("validation_train.txt");
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter train_writer = new BufferedWriter(fw);
		
		File file1 = new File("validation_test.txt");
		FileWriter fw1 = new FileWriter(file1.getAbsoluteFile());
		BufferedWriter test_writer = new BufferedWriter(fw1);
		
		File file2 = new File("test_labels.txt");
		FileWriter fw2 = new FileWriter(file2.getAbsoluteFile());
		BufferedWriter label_writer = new BufferedWriter(fw2);
		
		
		int counter = 1;
		for (; w < docs; w++) {
			if(docs/100!= 0 && w % (docs/100) == 0)
				System.out.println("reading training data, finished2 "+ (int) (w/(docs/100.0)) +"%");
			if ((l = input_file.readLine()) == null) break;
			if(w >= test_doc_num){
				train_writer.write(l);
				train_writer.newLine();
			}
			else{
				test_writer.write(counter+",0");
				counter++;
			    splits= l.split(" ");
			    for(int i = 1 ; i < splits.length; i ++){
			    	test_writer.write(" "+splits[i]);
			    }
			    label_writer.write(splits[0]);
			    label_writer.newLine();
			    test_writer.newLine();
			}
		}
        test_writer.close();
        train_writer.close();
        label_writer.close();
		return 0;
	}
	
	
	
}
