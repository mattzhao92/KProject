package tree;




import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.io.*;

import mtree.DistanceMetric;
import mtree.MTreeMap;
import mtree.Result;



class MyThread extends Thread
{
	private int startIndex, endIndex, workerId;
	private long startTime;
	ConcurrentHashMap<Integer, ArrayList<Integer>> result ;
	Double[][] test_matrix;
	Main main;
	public MyThread( int startIndex, int endIndex, ConcurrentHashMap<Integer, ArrayList<Integer>> result,
			Main main, int workerId){
		this.startIndex = startIndex;
		this.endIndex = endIndex;
		this.result = result;
		this.main = main;
		this.workerId = workerId;
		this.startTime = System.currentTimeMillis();
	}


	private List<Map.Entry<Integer, Double>> sortScoreMap(HashMap<Integer, Double> unsortMap) {

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


	@Override
	public void run()
	{
		
		
		for(int i = startIndex; i < endIndex; i ++)
		{
			if(((endIndex - startIndex)/100 != 0) &&
				(i - startIndex) % ((endIndex - startIndex)/100) == 0){
				long completedIn = System.currentTimeMillis() - this.startTime;
				startTime = System.currentTimeMillis();
				System.out.println(String.format("Worker[%d]", this.workerId) + " finished "+(i- startIndex) / ((endIndex - startIndex)/100) +"%" );
				System.out.println(String.format("Worker[%d]", this.workerId) +" time elapsed since last hundredth percent: "+completedIn);
			}
			Collection<Result<Bundle, Integer>> results = null;
			int N = 5;
			
			results = main.mTree.getNClosest(new Bundle(i, false) , N);
			//ArrayList<Integer> label = new ArrayList<Integer>();


			HashMap<Integer, Double> label_to_score = new HashMap<Integer, Double>();


			for (Result<Bundle, Integer> result : results) {       			
				for(int j = 0 ; j < main.getData().labels_train[result.key().doc].length; j ++){
					Integer label = main.getData().labels_train[result.key().doc][j];
					if (!label_to_score.containsKey(label)) {
						label_to_score.put(label,1.0);
					} else {
						label_to_score.put(label, label_to_score.get(label)+1.0);
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
					if (classified_labels.size() < 3 || Math.abs(firstEle.getValue() - currEle.getValue()) < 0.01) {
						classified_labels.add(currEle.getKey());
					}
				}
			}

			result.put(i, classified_labels);

		}
	}
}

public class Main {

	/** This DistanceMetric computes the distance between to (x,y) points. */
	static class PointMetric implements DistanceMetric<Bundle> {
		@Override
		public double distanceBtw(Bundle d1, Bundle d2) {
			return SparseData.calculate_distance(d1, d2) ;	
		}
	}


	public MTreeMap<Bundle, Integer> mTree;

	public MTreeMap<Bundle, Integer> getMTree(){
		return mTree;
	}


	private static SparseData data;

	public SparseData getData(){
		return data;
	}


	public void loadMTree(boolean useSerializedData) throws Exception{

		BufferedReader input_file= new BufferedReader(new FileReader("test-remapped.csv"));
		data = new SparseData(num_train_docs, num_test_docs);
		mTree = new MTreeMap<Bundle, Integer> (new PointMetric());
		CSVParser.read_libsvm_stream_test(data, num_test_docs,  input_file);
		System.out.println("Finished reading test");
		BufferedReader input_file_train= new BufferedReader(new FileReader("train-remapped.csv"));
		CSVParser.read_libsvm_stream(data, num_train_docs,  input_file_train);
		
		if (useSerializedData) {
			FileInputStream fileIn = new FileInputStream("./mTree.ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			mTree = (MTreeMap) in.readObject();
			in.close();
			fileIn.close();
		} else {
			for(int i = 0 ; i < num_train_docs; i ++){
				if(num_train_docs/100 != 0 && i% (num_train_docs/100) == 0)
					System.out.println("loading Mtree, finished "+i/(num_train_docs/100) +"%");
				mTree.put(new Bundle(i, true), i);
			}


			FileOutputStream fileOut =
					new FileOutputStream("./mTree.ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(mTree);
			out.close();
			fileOut.close();
			System.out.printf("Serialized data is saved in ./mTree.ser");

		}
		System.out.println("MTree is fully loaded");
	}

	//	static int num_test_docs  =  452167 ;
	//	static int num_train_docs = 2365436;

	static int num_test_docs  =  50;
	static int num_train_docs = 2365436;

	public static void main(String[] args) throws Exception {
		Main main = new Main();

		long time = System.currentTimeMillis();
		main.loadMTree(false);

		ConcurrentHashMap<Integer, ArrayList<Integer>> result = new ConcurrentHashMap<Integer, ArrayList<Integer>>();
		int num_thread = 32;
		MyThread[] threadArr = new MyThread[num_thread];
		for(int i = 0; i < num_thread ; i ++){
			if( i == num_thread -1){
				threadArr[i] = new MyThread ( i*(num_test_docs/num_thread),
						num_test_docs,  result, main, i);
			}
			else threadArr[i] = new MyThread ( i*(num_test_docs/num_thread),
					(i+1)*(num_test_docs/num_thread),  result, main, i);
		}
		for(int i = 0; i < num_thread ; i ++){
			threadArr[i].start(); 
		}
		for(int i = 0; i < num_thread ; i ++){
			threadArr[i].join();
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
		
		System.out.println("total time "+ (System.currentTimeMillis() - time));

	}
}