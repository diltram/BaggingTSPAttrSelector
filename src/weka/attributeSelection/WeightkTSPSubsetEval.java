package weka.attributeSelection;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import weka.attributeSelection.BaggingRanking.ResultObject;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

public class WeightkTSPSubsetEval extends TSPAbstractSubsetEvaluator {

    private static final long serialVersionUID = 5554640140623511699L;
    private double m_objsInRun;
    private int m_countObjects;
    private boolean m_debug;
    private Instances m_data;
    private double m_first_class = 0.0;
    private double m_second_class = 1.0;
    private BitSet m_instances;
    private int m_inst_first_class;
    private int m_inst_second_class;
    private List<ResultObject> m_result;

    public WeightkTSPSubsetEval() {
        resetOptions();
    }

    @Override
    public void buildEvaluator(Instances data) throws Exception {
        if (m_debug) {
            System.err.println("Starting building evaluator");
        }
        getCapabilities().testWithFail(data);

        m_data = new Instances(data);
        m_data.deleteWithMissingClass();
        setInstancesCount((int) (m_data.numInstances() * (getObjsInRun() / 100.0)));
    }

    @Override
    public double evaluateSubset(BitSet subset) throws Exception {
        if (m_debug) {
            System.err.println("Starting evaluating subset");
        }

        m_result = calculateScoreList(subset);

        if (m_debug) {
            System.err.println("Sorting results");
        }
        sortByScore(m_result);

        if (m_result.get(0).getScore() == m_result.get(1).getScore()) {
            if (m_debug) {
                System.err.println("Calculating average scores");
            }
            m_result = calculateAverageScoreList(subset);

            if (m_debug) {
                System.err.println("Sorting results");
            }
            sortByScore(m_result);
        }

        return 0;
    }

    private List<ResultObject> calculateAverageScoreList(BitSet subset) throws Exception {
        List<ResultObject> result = new ArrayList<>();
        double first_indicator, second_indicator;
        double first_probability, second_probability, score, avg_val;

        for (int i = subset.nextSetBit(0); i >= 0; i = subset.nextSetBit(i + 1)) {
            for (int j = subset.nextSetBit(1); j >= 0; j = subset.nextSetBit(j + 1)) {
                if (i == j) {
                    break;
                }
                avg_val = calculateAvgVal(i, j, m_instances);

                first_indicator = getAvgClassIndicator(i, j, m_instances, m_first_class, avg_val);
                second_indicator = getAvgClassIndicator(i, j, m_instances, m_second_class, avg_val);

                first_probability = getAvgProbability(first_indicator, m_inst_first_class);
                second_probability = getAvgProbability(second_indicator, m_inst_second_class);
                score = getScore(first_probability, second_probability);
                
                result.add(new ResultObject(i, j, score));
            }
        }

        return result;
    }

    private double getAvgProbability(double indicator, int instances_count) {        
        return indicator / instances_count;
    }

    private double getAvgClassIndicator(int i, int j, BitSet samples,
            double instance_class, double avg_val) {
        double indicator = 0;
        Instance instance;
        for (int n = samples.nextSetBit(0); n >= 0; n = samples.nextSetBit(n + 1)) {
            instance = m_data.instance(n);
            if (instance.classValue() == instance_class) {
                indicator += getAvgIndicator(instance, i, j, avg_val);
            }
        }
        return indicator;
    }

    private double getAvgIndicator(Instance instance, int i, int j, double avg_val) {
        double divided = 0;
        if (instance.value(i) != 0 && instance.value(j) != 0) {
            divided = instance.value(i) / instance.value(j);
            return divided / (avg_val + divided);
        }
        return 0;
    }

    private List<ResultObject> calculateScoreList(BitSet subset) {
        List<ResultObject> result = new ArrayList<>();
        double first_indicator, second_indicator;
        double first_probability, second_probability, score, avg_val;

        for (int i = subset.nextSetBit(0); i >= 0; i = subset.nextSetBit(i + 1)) {
            for (int j = subset.nextSetBit(1); j >= 0; j = subset.nextSetBit(j + 1)) {
                if (i == j) {
                    break;
                }
                avg_val = calculateAvgVal(i, j, m_instances);

                first_indicator = getClassIndicator(i, j, m_instances, m_first_class, avg_val);
                second_indicator = getClassIndicator(i, j, m_instances, m_second_class, avg_val);

                first_probability = getProbability(first_indicator, m_inst_first_class);
                second_probability = getProbability(second_indicator, m_inst_second_class);
                score = getScore(first_probability, second_probability);
                result.add(new ResultObject(i, j, score));
            }
        }

        return result;
    }

    public List<ResultObject> getResultList() {
        return m_result;
    }

    private double calculateAvgVal(int i, int j, BitSet instances) {
        double value = 0;
        Instance instance;

        for (int n = instances.nextSetBit(0); n >= 0; n = instances.nextSetBit(n + 1)) {
            instance = m_data.instance(n);
            value += instance.value(i) / instance.value(j);
        }
        return value / instances.cardinality();
    }

    private double getScore(double first_probability, double second_probability) {
        return Math.abs(first_probability - second_probability);
    }

    private double getProbability(double indicator, double instances_count) {
        return (1.0 / instances_count) * indicator;
    }

    private double getClassIndicator(int i, int j, BitSet samples,
            double instance_class, double avg_val) {
        double indicator = 0;
        Instance instance;
        for (int n = samples.nextSetBit(0); n >= 0; n = samples.nextSetBit(n + 1)) {
            instance = m_data.instance(n);
            if (instance.classValue() == instance_class) {
                indicator += getIndicator(instance, i, j, avg_val);
            }
        }
        return indicator;
    }

    private double getIndicator(Instance instance, int i, int j, double avg_val) {
        if ((instance.value(i) / instance.value(j)) < avg_val) {
            return 1.0;
        }
        return 0.0;
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<>(3);

        newVector.addElement(new Option("\tCount of top pairs.", "C", 100, "-C <top pairs>"));
        newVector.addElement(new Option("\tObjects per test in %.", "O", 66, "-O <objects percentage>"));
        newVector.addElement(new Option("\tOutput debugging info.", "D", 0, "-D"));

        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String optionString;
        resetOptions();

        optionString = Utils.getOption('O', options);
        if (optionString.length() != 0) {
            setObjsInRun(Integer.parseInt(optionString));
        }

        m_debug = Utils.getFlag('D', options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<>();

        options.add("-O");
        options.add("" + getObjsInRun());

        options.add("-D");
        options.add("" + getDebug());
        return options.toArray(new String[0]);
    }

    public void resetOptions() {
        setDebug(true);
        setObjsInRun(66);
    }

    public void setData(Instances data) {
        m_data = data;
        setClasses(m_data);
    }

    public Instances getData() {
        return m_data;
    }

    public boolean getDebug() {
        return m_debug;
    }

    public void setDebug(boolean debug) {
        m_debug = debug;
    }

    public double getObjsInRun() {
        return m_objsInRun;
    }

    public void setObjsInRun(double objsInRun) {
        m_objsInRun = objsInRun;
    }

    public int getCountObjects() {
        return m_countObjects;
    }

    public void setInstancesCount(int countObjects) {
        m_countObjects = countObjects;
        m_instances = new BitSet(m_countObjects);
//        Random random = m_data.getRandomNumberGenerator(1);
        Random random = new Random();

        while (m_instances.length() < m_countObjects) {
            m_instances.set(random.nextInt(m_countObjects), true);
        }

        m_inst_first_class = m_data.attributeStats(m_data.classIndex()).nominalCounts[0];
        m_inst_second_class = m_data.attributeStats(m_data.classIndex()).nominalCounts[1];
    }

    public void setClasses(Instances data) {
        List<Object> classes_list = Collections.list(data.classAttribute().enumerateValues());
    }

    /* Returns the capabilities of this evaluator.
     * 
     * @return the capabilities of this evaluator
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        return result;
    }
}
