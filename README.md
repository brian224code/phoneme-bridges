### Phoneme Bridges: Leveraging Phonetic Similarity for Low-Resource Language Understanding
Chris Ge, Brian Le, Daria Kryvosheieva | Final project for MIT 6.8611: Quantitative Methods for Natural Language Processing

#### Project Overview
Our project aims to improve language model performance on NLP tasks in **low-resource languages** (LRLs) through knowledge transfer from **high-resource languages** (HRLs). We especially focus on HRL-LRL pairs that share many **similar words in pronunciation** but use **different writing systems**. To enable knowledge transfer in this scenario, we use the **STILTs** finetuning method ([Phang et al., 2018](https://arxiv.org/pdf/1811.01088)) and augment our finetuning datasets with **romanizations**. We choose **mBERT** as an example model and **Hindi-Urdu** as an example HRL-LRL pair.

#### Our Pipeline
1. Pick an NLP task. We experiment with named entity recognition (NER) and part-of-speech (POS) tagging.
2. Gather a dataset for the task in each of the LRL and the HRL. Retrieve the romanizations of the two datasets’ input texts using a transliterator.
3. Fine-tune the language model on the NLP task in the HRL, randomly replacing a fixed proportion of words in the input text of the data by their romanizations.
4. Further fine-tune and evaluate the resulting model on the LRL task with both text and romanization.

(TODO: add image)

#### Experiments

For each of the PAN-X and UD-POS datasets, we fine-tune **four** versions of mBERT:
1. **mBERT<sub>text</sub>**: mBERT fine-tuned directly on the Urdu dataset (no romanizations);
2. **mBERT<sub>roman</sub>**: mBERT fine-tuned directly on the Urdu dataset (with romanizations concatenated);
3. **mBERT<sub>STILTs+text</sub>**: mBERT intermediately fine-tuned on the Hindi dataset, then further fine-tuned on the Urdu dataset (no romanizations);
4. **mBERT<sub>STILTs+roman</sub>**: mBERT intermediately fine-tuned on the Hindi dataset (with a quarter of the words replaced with romanizations), then further fine-tuned on the Urdu dataset (with romanizations concatenated).

(TODO: add image)

#### Results

Table 1 shows the performance of our models (measured as macro-F1 score) on the two tasks, and Table 2 shows the results of our statistical significance test (paired bootstrap resampling). Overall, our method yielded **improvement**, but it was **not statistically significant**.

| Model | POS Tagging Score | NER Score |
|-------|-------------------|-----------|
| mBERT<sub>text</sub> | 0.8700 | 0.9770 |
| mBERT<sub>roman</sub> | 0.8728 | 0.9780 |
| mBERT<sub>STILTs+text</sub> | 0.8702 | 0.9763 |
| mBERT<sub>STILTs+roman</sub> | 0.8735 | 0.9788 |

Table 1: Our results (macro-F1 scores) for POS tagging and NER.

<table>
    <thead>
        <tr>
            <th>Comparison</th>
            <th>Task</th>
            <th>Mean Difference</th>
            <th>95% Confidence Interval</th>
            <th>P-value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>mBERT<sub>roman</sub> - mBERT<sub>text</sub></td>
            <td>POS</td>
            <td>0.0029</td>
            <td>[-0.0016, 0.0071]</td>
            <td>0.2180</td>
        </tr>
        <tr>
            <td>NER</td>
            <td>0.0010</td>
            <td>[-0.0066, 0.0079]</td>
            <td>0.7680</td>
        </tr>
        <tr>
            <td rowspan=2>mBERT<sub>STILTs+roman</sub> - mBERT<sub>STILTs+text</sub></td>
            <td>POS</td>
            <td>0.0035</td>
            <td>[-0.0019, 0.0092]</td>
            <td>0.2200</td>
        </tr>
        <tr>
            <td>NER</td>
            <td>0.0026</td>
            <td>[-0.0038, 0.0095]</td>
            <td>0.4100</td>
        </tr>
    </tbody>
</table>

Table 2: Statistical test results.
