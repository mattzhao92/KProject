import java.io.*;
import java.util.*;


public class HierarchyLoader {
	static int num_test_docs  = 452167;
	static int num_train_docs =  2365436;
	public static void main(String[] args) throws Exception{
		SparseData data = new SparseData(num_train_docs, num_test_docs);
		BufferedReader input_file_train= new BufferedReader(new FileReader("train-remapped.csv"));
		ValidationFileMaker.read_libsvm_stream(data,num_train_docs,  input_file_train);
		/*
		HierarchyLoader loader = new HierarchyLoader();
	    loader.loadHierarchyFile();
	    ArrayList<ArrayList<Integer>> lastLayer = loader.getLastLayer();
	    
	    
	    File file = new File("lastLayer.txt");

		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		
		for(int i = 0; i < lastLayer.size(); i ++){
			for(int j = 0; j < lastLayer.get(i).size(); j ++){
				bw.write(lastLayer.get(i).get(j)+" "+i+"\n");
			}
		}
	    bw.close();
	    */
	    /*
		SparseData data = new SparseData(num_train_docs, num_test_docs);
		BufferedReader input_file_train= new BufferedReader(new FileReader("train-remapped.csv"));
		int readNum = read_libsvm_stream(data,num_train_docs,  input_file_train);
		System.out.println("READ NUM "+readNum)	;
		//ArrayList<Integer> allLeaf = new ArrayList<Integer>();
		HashSet<Integer> allLeaf = new HashSet<Integer>();
		for(ArrayList<Integer> arr : lastLayer){
			allLeaf.addAll(arr);
		}
		
		// get a file : label -> num of train docs that has that label
        HashMap<Integer, Integer> countMap = new HashMap<Integer,Integer>();
	    int d1 = data.labels_train.length;
		for(int i = 0; i < d1 ; i ++){
			  	for(int j = 0; j < data.labels_train[i].length;  j ++){
			  		int temp = data.labels_train[i][j];
			  	     if(countMap.containsKey(temp)){
			  	    	 countMap.put(temp, countMap.get(temp)+1);
			  	     }
			  	     else countMap.put(temp, 1);
			  	}
		}
		File file = new File("labelCount.txt");

		int counter = 0;
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		for(Integer label : countMap.keySet()){
			if(allLeaf.contains(label))
			    fw.write(label+" "+countMap.get(label)+"\n");
			else{
				counter ++;
			}
		}
		
		System.out.println("counter is "+counter +"total is "+ allLeaf.size());
		bw.close();	
		*/
	}
	
    HashMap<Integer,ArrayList<Integer>> graph = new HashMap<Integer,ArrayList<Integer>>();
	private void loadHierarchyFile() throws Exception{
		BufferedReader br = new BufferedReader(new FileReader("hierarchy.txt"));
	    String line = br.readLine();
	    while (line != null) {
	          String[] numPair = line.split(" ");
	          Integer int1 = Integer.parseInt(numPair[0]);
	          Integer int2 = Integer.parseInt(numPair[1]);
              if(graph.containsKey(int1)){
            	  graph.get(int1).add(int2);
              }
              else{
            	  graph.put(int1, new ArrayList<Integer>());
            	  graph.get(int1).add(int2);
              }
	          line = br.readLine();
	    }
        br.close();
	}
	
	private ArrayList<ArrayList<Integer>> getLastLayer(){
		ArrayList<ArrayList<Integer>> returnResult = new ArrayList<ArrayList<Integer>>();
		for(Integer int1: graph.keySet()){
			int flag = 1;
			for(Integer int2 : graph.get(int1)){
				if(graph.containsKey(int2)){
					flag = 0;
				}
			}
			if(flag == 1){
				returnResult.add(graph.get(int1));
			}
		}
		return returnResult;
	}
	
	public static int read_libsvm_stream(SparseData data, int docs, BufferedReader input_file) throws Exception {
		String l;
		String[] splits, s;
		int[] labels, terms;
		int[] counts = null;
		int w= 0;
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
