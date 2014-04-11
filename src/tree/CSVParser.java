package tree;

import java.io.BufferedReader;
import java.util.Arrays;

public class CSVParser {

	public static void read_top_features_test(SparseData data, int docs, BufferedReader br) throws Exception {
		String sCurrentLine;

		for (int i = 0; i < docs; i++) {
			sCurrentLine = br.readLine();
			String [] splits = sCurrentLine.split(" ");
			for (int j = 1; j < splits.length; j++) {
				//int  a = data.top_features_test[i][j-1];
				data.top_features_test[i][j-1] = Integer.parseInt(splits[j]);	
			}
		}
	}
	
	public static int read_libsvm_stream_test(SparseData data, int docs, BufferedReader input_file) throws Exception {
		String l;
		String[] splits, s;
		int[] terms;
		int[] counts = null;
		int w= 0;
		for (; w < docs; w++) {
			if(docs/100 !=0 && w % (docs/100) == 0)
				System.out.println("reading testing data, finished "+w/(docs/100) +"%");
			if ((l = input_file.readLine()) == null) break;
			int term_c= 0;//, length= 0;
			for (char c: l.toCharArray()) if (c==':') term_c++;
			splits= l.split(" ");			
			SparseData.terms_test[w]= terms= new int[term_c];
			data.counts_test[w]= counts= new int[term_c];
			for (int i=0; i < Math.min(SparseData.max_num_features, splits.length-1) ; ) {
				//System.out.println(splits[i]);
				s= splits[i+1].split(":");
				Integer term= Integer.decode(s[0]);
				terms[i]= term;
				// if(i % 10 ==0 ) System.out.println(term);
				counts[i++]= Integer.parseInt(s[1]);
				//data.addCount(term);
			}
		}
		if (w != docs) data.doc_count_test = w;
		return w;
	}
	

	public static int read_libsvm_stream(SparseData data, int docs, BufferedReader input_file) throws Exception {
		String l;
		String[] splits, s;
		int[] labels, terms;
		int[] counts = null;
		int w= 0;
		input_file.readLine(); // skip first line
		for (; w < docs; w++) {
			if(docs/100!= 0 && w % (docs/100) == 0)
				System.out.println("reading training data, finished2 "+ (int) (w/(docs/100.0)) +"%");
			if ((l = input_file.readLine()) == null) break;
			int term_c= 0, i= 0;//, length= 0;
			for (char c: l.toCharArray()) if (c==':') term_c++;
			splits= l.split("[\\s\\,]+");
			int label_c= splits.length - term_c;
			data.labels_train[w]= labels= new int[label_c];
			SparseData.terms_train[w]= terms= new int[term_c];
			data.counts_train[w]= counts= new int[term_c];
			for (; i < label_c; i++) {
				s= splits[i].split(",")[0].split(";");
				labels[i]= Integer.decode(s[0]);
			}
			
			
			for ( ; i < Math.min(SparseData.max_num_features, splits.length) ; ) {
				//System.out.println(splits[i]);
				s= splits[i].split(":");
				Integer term= Integer.decode(s[0]);
				terms[i - label_c]= term;
				//counts[i++ - label_c]= (float)Integer.decode(s[1]);
				counts[i++ - label_c]= Integer.parseInt(s[1]);
				data.addCount(term);
			}
		}
		if (w != docs) data.doc_count_train = w;
		return w;
	}

}
