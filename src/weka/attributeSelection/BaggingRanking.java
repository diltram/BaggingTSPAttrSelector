package weka.attributeSelection;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

public class BaggingRanking extends ASSearch implements OptionHandler {

	private static final long serialVersionUID = -6622671521414498563L;
	private int m_runCount;
	private boolean m_returnAttrs;
	private double m_attrsInRun;
	private int m_objsInRun;
	private int m_topAttrs;
	private boolean m_debug;
	
	public BaggingRanking() {
		resetOptions();
	} 

	@Override
 	public int[] search(ASEvaluation ASEval, Instances data) throws Exception {
		int numAttribs = data.numAttributes();
		int position;
                BitSet selectedAttrs = new BitSet(numAttribs);
		m_attrsInRun = (int) Math.sqrt(numAttribs); // pomyśleć nad sposobem wybierania ilości elementów
                Random rand = new Random();
		Random randomGen = data.getRandomNumberGenerator(rand.nextLong());
		List<List<ResultObject>> results = new ArrayList<>();
		TSPSubsetEval ASEvaluator = (TSPSubsetEval) ASEval;
		
		for(int i = 0; i < m_runCount; i++) {
			if(m_debug) {
				System.err.println("Iteration no. " + (i + 1));
			}
			BitSet newSet = new BitSet(numAttribs);
			int count = 0;
                        
//                        if((numAttribs - selectedAttrs.size()) < m_attrsInRun) {
//                            m_runCount = i - 1;
//                            break;
//                        }
                        
			while(count < m_attrsInRun) {
				position = randomGen.nextInt(numAttribs);
				if(position != data.classIndex() && !newSet.get(position) && !selectedAttrs.get(position)) {
					newSet.set(position, true);
                                        selectedAttrs.set(position, true);
					count++;
				}
			}
			ASEvaluator.evaluateSubset(newSet);
			results.add(ASEvaluator.getResultList().subList(0, m_topAttrs));
		}
		
		int[] rank = rankResults(results, data.numAttributes(), m_topAttrs);
		return rank;
	}

	private int[] rankResults(List<List<ResultObject>> results, int attrs, int limit) {
		List<ResultObject> one_result;
		AttributeRank[] rank = new AttributeRank[attrs];
		double log = Math.log(0.3);
		double to_positive = Math.abs(calculateRank(log, attrs + 1));
		double value;
		int attr;
		
		for(int i = 0; i < results.size(); i++) {
			one_result = results.get(i);
			for(int j = 0; j < one_result.size(); j++) {
				value = calculateRank(log, (j + 1)) + to_positive;
				
				attr = one_result.get(j).getI();
				if(rank[attr] == null) {
					rank[attr] = new AttributeRank(attr, value);
				} else {
					rank[attr].addRanking(value);	
				}

				attr = one_result.get(j).getJ();
				if(rank[attr] == null) {
					rank[attr] = new AttributeRank(attr, value);
				} else {
					rank[attr].addRanking(value);	
				}
			}
		}
		
		for(int i = 0; i < rank.length; i++) {
			if(rank[i] == null) {
				rank[i] = new AttributeRank(i, -Double.MAX_VALUE);
			}
		}
		
		List<AttributeRank> attribsList = Arrays.asList(rank);
		
		Collections.sort(attribsList, new Comparator<AttributeRank>() {
			@Override
			public int compare(AttributeRank o1, AttributeRank o2) {
				if(o1.getRank() < o2.getRank()) {
					return 1;
				} else if(o1.getRank() == o2.getRank()) {
                                    return 0;
                                }
				return -1;
			}
		});
		
		return getArrayOfTop(attribsList, limit);
	}

	private int[] getArrayOfTop(List<AttributeRank> attribsList, int limit) {
		int[] result = new int[limit];

		for(int i = 0; i < limit; i++) {
			result[i] = attribsList.get(i).getAttribute();
		}

		return result;
	}

	private double calculateRank(double d, int i) {
		return Math.log(i) / d;
	}

	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<>(4);

		newVector.addElement(new Option("\tSpecify how many times should be started algorythm.", "R", 100, "-R <run count>"));
		newVector.addElement(new Option("\tShould return attributes.", "RA", 1, "-RA <1 | 0>"));
		newVector.addElement(new Option("\tAmount of top attributes to return", "TA", 100, "-TA <num>"));
		newVector.addElement(new Option("\tSize of lookup cache for evaluated subsets." + "\n\tExpressed as a multiple of the number of" 
										+ "\n\tattributes in the data set. (default = 1)", "S", 1, "-S <num>"));

		return newVector.elements();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String optionString;
	    resetOptions();

	    optionString = Utils.getOption("R", options);
	    if (optionString.length() != 0) {
	      setRunCount(Integer.parseInt(optionString));
	    }

	    optionString = Utils.getOption("TA", options);
	    if (optionString.length() != 0) {
	      setTopAttrs(Integer.parseInt(optionString));
	    }
	    
	    m_debug = Utils.getFlag("D", options);
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> options = new Vector<String>();
		options.add("-R");
		options.add("" + getRunCount());

		options.add("-TA");
		options.add("" + getTopAttrs());

		options.add("-OR");
		options.add("" + getObjsInRun());

		options.add("-RET");
		options.add("" + getReturnAttrs());
		
		options.add("-D");
		options.add("" + getDebug());
		
		return options.toArray(new String[0]);
	}

	private void resetOptions() {
		this.setRunCount(100);
		this.setAttrsInRun(0.66);
		this.setObjsInRun(m_objsInRun);
		this.setReturnAttrs(true);
		this.setTopAttrs(100);
		this.setDebug(true);
	}

	public int getRunCount() {
		return m_runCount;
	}

	public void setRunCount(int runCount) {
		m_runCount = runCount;
	}

	public boolean getReturnAttrs() {
		return m_returnAttrs;
	}

	public void setReturnAttrs(boolean returnAttrs) {
		m_returnAttrs = returnAttrs;
	}

	public double getAttrsInRun() {
		return m_attrsInRun;
	}

	public void setAttrsInRun(double attrsInRun) {
		m_attrsInRun = attrsInRun;
	}

	public int getObjsInRun() {
		return m_objsInRun;
	}

	public void setObjsInRun(int objsInRun) {
		m_objsInRun = objsInRun;
	}

	public int getTopAttrs() {
		return m_topAttrs;
	}

	public void setTopAttrs(int retTopAttrs) {
		m_topAttrs = retTopAttrs;
	}
	
	public boolean getDebug() {
		return m_debug;
	}
	
	public void setDebug(boolean debug) {
		m_debug = debug;
	}
	
	public static class ResultObject implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = 7566991890437481535L;
		private int m_i;
		private int m_j;
		private double m_score;
		
		public ResultObject(int i, int j, double score) {
			m_i = i;
			m_j = j;
			m_score = score;
		}
		
		public int getI() {
			return m_i;
		}
		
		public int getJ() {
			return m_j;
		}
		
		public double getScore() {
			return m_score;
		}
		
		public String toString() {
			return "i = " + m_i + ", j = " + m_j + ", score = " + m_score;
		}
	}
	
	public static class AttributeRank implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = 4035093188592742478L;
		private int m_attribute;
		private double m_rank;
		
		public AttributeRank(int attribute, double rank) {
			m_attribute = attribute;
			m_rank = rank;
		}
		
		public int getAttribute() {
			return m_attribute;
		}
		
		public double getRank() {
			return m_rank;
		}
		
		public void addRanking(double rank) {
			m_rank += rank;
		}
	}
}