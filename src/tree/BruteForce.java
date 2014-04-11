package tree;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
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


public class BruteForce {
	static AtomicInteger counter = new AtomicInteger();
	static SparseData data;
	static int num_test_docs  = 452167;
	static int num_train_docs =  2365436;
	static int num_workers = 32;
	static ConcurrentHashMap<Integer, ArrayList<Integer>> result ;

	
	static int k_nearest_neighbor = 5;
	
	
	private static void clear_distances(double [] distances) {
		for (int i = 0; i < distances.length; i++) {
			distances[i] = Double.MAX_VALUE;
		}
	}
	
	private static void  clear_doc_indices(int [] doc_indices) {
		for (int i = 0; i < doc_indices.length; i++) {
			doc_indices[i] = -1;
		}
	}
	
	
	private static void update_best(double [] distances, int [] doc_indices, double new_dist, int new_index) {
		
		int i;
		for (i = 0; i < distances.length; i++) {
			if (new_dist < distances[i]) {
				break;
			}
		}
		
		int j = distances.length - 1;
		
		while (j > i) {
			distances[j] = distances[j-1];
			doc_indices[j] = doc_indices[j-1];
			j --;
		}
		
		if (i < distances.length) {
			distances[i] = new_dist;
			doc_indices[i] = new_index;	
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
	
	static int chunk_size = 5000;
	
	
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

		public KaggleWorkerThread(ArrayList<Tuple> q, int workerIdIn) { 
			queue = q; 
			this.workerId = workerIdIn;
			System.out.println(String.format("Worker[%d]", this.workerId) +" initialized");

		}

		public void run() {
			try {
				
				while (true) {
					int count = counter.getAndIncrement();
					if (count < queue.size()) {
						Tuple t = queue.get(count);
						serve(t);
					} else {
						break;
					}
				}
				
				
			
			} catch (Exception e) { 
				e.printStackTrace();
			}
		}

		public void serve(Tuple tuple) {
			double [] distances = new double[k_nearest_neighbor];
			int []    doc_indcies = new int [k_nearest_neighbor];
			
//			double [] tfidf_distances = new double[num_top_tfidf_features];
//			int [] features = new int[num_top_tfidf_features];
//			
			
			int start = tuple.start;
			int end = tuple.end;
			
			System.out.println(String.format("Worker[%d]", this.workerId) + " Processing ["+start +", "+end+"]");
	

			for (int i = start; i < end; i++) {
				Bundle test_doc = new Bundle(i, false);
				clear_distances(distances);
				clear_doc_indices(doc_indcies);
				
//				clear_distances(tfidf_distances);
//				clear_doc_indices(features);
//				
//				// get top tfidf features from this set
//				for(int j = 0 ; j < data.terms_test[i].length; j ++){
//					int term = data.terms_test[i][j];
//					double tfidf_distance = - data.gettfidf(term, test_doc);
//					//System.out.println(">>>> "+tfidf_distance);
//					update_best(tfidf_distances, features, tfidf_distance, term);
//					//System.out.println("tfidf_distances "+j +" "+ Arrays.toString(tfidf_distances));
	//
//				}

				int [] features = data.top_features_test[i];
				for (int j = 0; j < num_train_docs; j++) {
					Bundle train_doc = new Bundle(j, true);
					
					// check whether the top features present in the current training doc or not.
					int num_top_features = 0;
					
					for (int t1 = 0; t1 < 3; t1++) {
						int top_term = features[t1];
						for (int t2 = 0; t2 < data.terms_train[j].length; t2 ++) {
							if (top_term == data.terms_train[j][t2]) {
								num_top_features++;
							}
						}
					}

					if (num_top_features >= 1) {
						double distance = SparseData.calculate_distance(test_doc, train_doc);
						update_best(distances, doc_indcies, distance, j);
					}
				}
				
				//System.out.println("doc_indices "+ Arrays.toString(doc_indcies));

				HashMap<Integer, Double> label_to_score = new HashMap<Integer, Double>();


				for (int doc_indice : doc_indcies) {      
					if (doc_indice != -1) {
						for(int j = 0 ; j < data.labels_train[doc_indice].length; j ++){
							Integer label = data.labels_train[doc_indice][j];
							if (!label_to_score.containsKey(label)) {
								label_to_score.put(label,1.0);
							} else {
								label_to_score.put(label, label_to_score.get(label)+1.0);
							}
						}        		
					}
				}
				
				List<Map.Entry<Integer, Double>> sortedList = sortScoreMap(label_to_score);
				assert(sortedList.size() == label_to_score.size());
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

				//System.out.println(">>> labels: "+ Arrays.toString(classified_labels.toArray()));


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
		BufferedReader input_file_train= new BufferedReader(new FileReader("train-remapped.csv"));
		CSVParser.read_libsvm_stream(data,num_train_docs,  input_file_train);
		
		data.compute_test_Idf();
		
		
		
		ArrayList<KaggleWorkerThread> workers = new ArrayList<KaggleWorkerThread>();
		
		for (int n = 0; n < num_workers; n++) {
			KaggleWorkerThread worker = new KaggleWorkerThread(queue,n);
			workers.add(worker);
			worker.start();
		}
		
		
		for (int n = 0; n < num_workers; n++) {
			workers.get(n).join();
		}
		
		
		
		File file = new File("result.txt");
		file.createNewFile();
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		bw.write("Id,Predicted");
		bw.newLine();

		System.out.println("result size is "+result.size());

		for(int i = 0; i < num_test_docs; i ++){
			bw.write((i+1) +"," );
			ArrayList<Integer> temp = result.get(i);
			for(Integer integer: temp){
				bw.write(integer+" ");
			}
			bw.newLine();
		}
		bw.close();
		
	}
}
