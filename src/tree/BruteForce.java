package tree;

import java.io.BufferedReader;
import java.io.FileReader;


public class BruteForce {

	static SparseData data;
	static int num_test_docs  = 20000;
	static int num_train_docs = 2365436;
	
	public static void main(String [] args) throws Exception{
		BufferedReader input_file= new BufferedReader(new FileReader("test-remapped.csv"));
		data = new SparseData(num_train_docs, num_test_docs);

		CSVParser.read_libsvm_stream_test(data, num_test_docs,  input_file);
		System.out.println("Finished reading test");
		BufferedReader input_file_train= new BufferedReader(new FileReader("train-remapped.csv"));
		CSVParser.read_libsvm_stream(data,num_train_docs,  input_file_train);
		
		
	}
}
