
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;


public class Validator {
	static AtomicInteger counter = new AtomicInteger();
	static SparseData data;
	static int num_test_docs  = 500000;
	static int num_train_docs =  1865436;
	//static int num_test_docs  = 100;
	//static int num_train_docs =  100;
	static int num_workers = 1;
	static ConcurrentHashMap<Integer, ArrayList<Integer>> result ;

	static int k_nearest_neighbor = 10;
	
	
	private static void clear_distances(double [] distances) {
		for (int i = 0; i < distances.length; i++) {
			distances[i] = Double.MAX_VALUE;
		}
	}
	

	private static List<Map.Entry<Integer, Double>> sortScoreMap(HashMap<Integer, Double> unsortMap) {

		List<Map.Entry<Integer, Double>> list = new LinkedList<Map.Entry<Integer, Double>>(unsortMap.entrySet());

		// sort list based on comparator
		Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
			@Override
			public int compare(Entry<Integer, Double> o1,
					Entry<Integer, Double> o2) {
				return (o2).getValue()
						.compareTo((o1).getValue());
			}
		});

		return list;
	}
	
	static int num_top_tfidf_features = 3;
	
	static int chunk_size = 50;
	
	
	static class Tuple {
		int start;
		int end;
		
		public Tuple (int startIn, int endIn) {
			this.start = startIn;
			this.end = endIn;
		}
	}
	
	private final static ArrayList<Tuple> queue = new ArrayList<Tuple>();
	
	private static void populateBoundedQueue() throws Exception {
		
		Tuple tuple = null;
		for (int i = 0; i < num_test_docs / chunk_size; i++) {
			tuple = new Tuple(i * chunk_size, Math.min((i+1) * chunk_size, num_test_docs));
			queue.add(tuple);
		}
		
		//if (tuple != null)
			tuple.end = num_test_docs;
	}
	
	static class KaggleWorkerThread extends Thread {
		private final ArrayList<Tuple> queue;
		private final int workerId;
        private double[] params;
		public KaggleWorkerThread(ArrayList<Tuple> q, int workerIdIn,double[] params) { 
			queue = q; 
			this.workerId = workerIdIn;
			//System.out.println(String.format("Worker[%d]", this.workerId) +" initialized");
            this.params = params;
		}

		public void run() {
			try {
				
				while (true) {
					int count = counter.getAndIncrement();
					if (count < queue.size()) {
						Tuple t = queue.get(count);
						serve(t,params);
					} else {
						break;
					}
				}
				
				
			
			} catch (Exception e) { 
				e.printStackTrace();
			}
		}

		public void serve(Tuple tuple, double[] params) {
			double [] distances = new double[k_nearest_neighbor];
			int start = tuple.start;
			int end = tuple.end;
			
			//System.out.println(String.format("Worker[%d]", this.workerId) + " Processing ["+start +", "+end+"]");
	

			for (int i = start; i < end; i++) {
				clear_distances(distances);
                
                // first contribution measurement
				HashMap<Integer, Double> label_to_score_first = new HashMap<Integer, Double>();
                int[] doc_indcies = data.five_neighbors[i];         
                double weight = 1;
				for (int jj = 0; jj < k_nearest_neighbor;jj ++) {    
					int doc_indice = doc_indcies[jj];
					if (doc_indice != -1) {
						for(int j = 0 ; j < data.labels_train[doc_indice].length; j ++){
							Integer label = data.labels_train[doc_indice][j];
							if (!label_to_score_first.containsKey(label)) {
								label_to_score_first.put(label,1.0);
							} else {
								label_to_score_first.put(label, label_to_score_first.get(label)+1.0*weight);
							}
						}        		
					}
					
				}
				
				// second contribution measurement: label count
				HashMap<Integer, Double > label_to_score_second = new HashMap<Integer, Double>();
				for(Integer label : label_to_score_first.keySet() ){
					if(data.label_count.containsKey(label)){
						label_to_score_second.put(label, 1.0*data.label_count.get(label));
					}
				}
				
				// thrid contribution measurement:  last layer
				HashMap<Integer, Double > label_to_score_third = new HashMap<Integer, Double>();

				HashMap<Integer, ArrayList<Integer>> hierarchy_measure = new HashMap<Integer, ArrayList<Integer>>();
				for(Integer label : label_to_score_first.keySet() ){
					Integer nodeIndex = data.last_layer.get(label);
					if(nodeIndex != null){
						if(hierarchy_measure.containsKey(nodeIndex)){
							hierarchy_measure.get(nodeIndex).add(label);
						}
						else{
							hierarchy_measure.put(nodeIndex, new ArrayList<Integer>());
							hierarchy_measure.get(nodeIndex).add(label);
						}
					}
				}
				// get the list of labels that has the most members in the same node
				ArrayList<Integer> mostArr = new ArrayList<Integer>();
				for(ArrayList<Integer> arr: hierarchy_measure.values()){
					if(arr.size() > mostArr.size()){
						mostArr = arr;
					}
				}
                for(Integer label: mostArr){
                	label_to_score_third.put(label, 1.0);
                }
				
                /*
                HashMap<Integer, Double> label_to_score_second = new HashMap<Integer, Double>();
				HashMap<Integer, Double> label_to_num= new HashMap<Integer, Double>();

                int[] doc_indices = data.five_neighbors[i]; 
                double[] doc_distances = new double[k_nearest_neighbor];
                int counter = 0;
                
				for (int doc_indice : doc_indcies) { 
					double distance = data.cosine_similarity(new Bundle(doc_indice,true	), new Bundle(i,false));
					for(int j = 0 ; j < data.labels_train[doc_indice].length; j ++){
					     if(label_to_score_second.keySet().contains(data.labels_train[doc_indice][j])){
					    	 label_to_score_second.put(data.labels_train[doc_indice][j], 
					    			 (label_to_score_second.get(data.labels_train[doc_indice][j])*
					    			 label_to_num.get(data.labels_train[doc_indice][j])+distance)/
					    			 (1+label_to_num.get(data.labels_train[doc_indice][j])));
					    	 label_to_num.put(data.labels_train[doc_indice][j], label_to_num.get(data.labels_train[doc_indice][j])+1);
					     }
					     else{
					    	 label_to_num.put(data.labels_train[doc_indice][j], 1.0);
					    	 label_to_score_second.put(data.labels_train[doc_indice][j], distance);
					     }
					}
				}
				*/
                // now sum the three score maps
                for(Integer label:label_to_score_second.keySet()){
               // 	System.out.println("first "+label_to_score_first.get(label)+
               // 			" second "+label_to_score_second.get(label)+" third "+label_to_score_third.get(label));
                	
                	label_to_score_first.put(label, label_to_score_first.get(label)+  params[1]*label_to_score_second.get(label) );
                }
                
                for(Integer label:label_to_score_third.keySet()){
                	label_to_score_first.put(label, label_to_score_first.get(label)+  params[0]*label_to_score_third.get(label) );
                }                
				
				List<Map.Entry<Integer, Double>> sortedList = sortScoreMap(label_to_score_first);
				assert(sortedList.size() == label_to_score_first.size());
				ArrayList<Integer> classified_labels = new ArrayList<Integer>();

				if (sortedList.size() == 0) {
					classified_labels.add(100000);
				} else {
					Map.Entry<Integer, Double> firstEle = sortedList.get(0);
					for (int j = 0; j < sortedList.size(); j++) {
						Map.Entry<Integer, Double> currEle = sortedList.get(j);
						if ((classified_labels.size() < 3 || Math.abs(firstEle.getValue() - currEle.getValue()) < 0.01) &&
								classified_labels.size() < 4){
							classified_labels.add(currEle.getKey());
						}
					}
				}				
				result.put(i, classified_labels);
			}
		}
	}
	
	public static void main(String [] args) throws Exception{
		populateBoundedQueue();		
		BufferedReader input_file= new BufferedReader(new FileReader("test-remapped.csv"));
		data = new SparseData(num_train_docs, num_test_docs);
		result = new ConcurrentHashMap<Integer, ArrayList<Integer>>();		
		BufferedReader input_topfeatures= new BufferedReader(new FileReader("tftopfeatures.txt"));
		CSVParser.read_top_features_test(data, num_test_docs, input_topfeatures);		
		CSVParser.read_libsvm_stream_test(data, num_test_docs,  input_file);
		System.out.println("Finished reading test");
		BufferedReader input_file_train= new BufferedReader(new FileReader("temp/train-remapped.csv"));
		CSVParser.read_libsvm_stream(data,num_train_docs,  input_file_train);	
		data.compute_test_Idf();
		data.load_knn("temp/knn.txt",10);
		data.load_label_count();
		data.load_last_layer();
		double params[][] ={
//				{-0.000003,-0.00005},
      
				{10.0,-0.000003},
				{1.0,-0.000003},
				{0.1,-0.000003},
				{0.01,-0.000003},
				{0,-0.000003},
				{-0.000001,-0.000003},
				{-0.000003,-0.000003},
				{-0.000005,-0.000003},
				{-0.00001,-0.000003},
				{-0.0001,-0.000003},
				{-0.001,-0.000003},
				{-0.01,-0.000003},
				{-0.1,-0.000003},
				{-10,-0.000003},
				
				/*
				{1,0},
				{2,0},
				
				{-0.01,-0.1},
				{-0.001,-0.1},
				{-0.0001,-0.1},
				{-0.1,-0.01},
				{-0.01,-0.01},
				{-0.001,-0.01},
				{-0.0001,-0.01},
				{-0.1,-0.0001},
				{-0.01,-0.0001},
				{-0.001,-0.0001},
				
				{-0.0001,-0.0001},
				{-0.00001,-0.0001},
				
				{-0.000001,-0.0001},
				
				
				{-0.000003,-0.0002},
				{-0.000003,-0.0001},
				{-0.000003,-0.00005},
				{-0.000003,-0.0008},
				{-0.000003,-0.0010},
				{-0.000003,-0.0012},
*/

		};
		int maxAccuracy = 0;
		int maxIndex = 0;
		for(int param_index = 0; param_index < params.length; param_index++){
			ArrayList<KaggleWorkerThread> workers = new ArrayList<KaggleWorkerThread>();		
			for (int n = 0; n < num_workers; n++) {
				KaggleWorkerThread worker = new KaggleWorkerThread(queue,n,params[param_index]);
				workers.add(worker);
				worker.start();
			}
			for (int n = 0; n < num_workers; n++) {
				workers.get(n).join();
			}
			counter = new AtomicInteger();

		    BufferedReader br = new BufferedReader(new FileReader("test_labels.txt"));
			String line;
			int current_counter = 0;
			int accuracyCounter = 0;
			while ((line = br.readLine()) != null) {
				String[] splits = line.split(",");
				for(String str: splits){
					if(result == null) System.out.println("result is null");
					if(result.size() == 0)  System.out.println("result is zero");
				    if( result.get(current_counter).contains(Integer.parseInt(str))){
				    	accuracyCounter ++;
				    	break;
				    }
				}
				current_counter ++;
			}
			br.close();
			System.out.println(params[param_index][0]+" "+params[param_index][1]+" "+accuracyCounter);
			if(accuracyCounter > maxAccuracy){
				maxAccuracy = accuracyCounter;
				maxIndex = param_index;
			}
			result.clear();
		}
		//System.out.println("max index is "+maxIndex);
		/*
		File file = new File("result.txt");
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		bw.write("Id,Predicted");
		bw.newLine();

		System.out.println("result size is "+result.size());

		for(int i = 0; i < num_test_docs; i ++){
			bw.write((i+1) +"," );
			ArrayList<Integer> temp = result.get(i);
			for(int j = 0; j < temp.size(); j ++){
				bw.write(temp.get(j)+"");
				if(j != temp.size()-1) bw.write(" ");
			}
			if(i != num_test_docs-1) bw.newLine();
		}
		bw.close();
		*/
	}
}
