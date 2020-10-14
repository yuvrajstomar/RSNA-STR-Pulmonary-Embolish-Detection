# RSNA-STR-Pulmonary-Embolish-Detection
**kaggle competition event**

If every breath is strained and painful, it could be a serious and potentially life-threatening condition. A pulmonary embolism (PE) is caused by an artery blockage in the lung. It is time consuming to confirm a PE and prone to overdiagnosis. Machine learning could help to more accurately identify PE cases, which would make management and treatment more effective for patients.

Currently, CT pulmonary angiography (CTPA), is the most common type of medical imaging to evaluate patients with suspected PE. These CT scans consist of hundreds of images that require detailed review to identify clots within the pulmonary arteries. As the use of imaging continues to grow, constraints of radiologists’ time may contribute to delayed diagnosis.

The Radiological Society of North America (RSNA®) has teamed up with the Society of Thoracic Radiology (STR) to help improve the use of machine learning in the diagnosis of PE.

In this competition, you’ll detect and classify PE cases. In particular, you'll use chest CTPA images (grouped together as studies) and your data science skills to enable more accurate identification of PE. If successful, you'll help reduce human delays and errors in detection and treatment.

With 60,000-100,000 PE deaths annually in the United States, it is among the most fatal cardiovascular diseases. Timely and accurate diagnosis will help these patients receive better care and may also impove outcomes.

*****Acknowledgments***

The Radiological Society of North America (RSNA®) is an international society of radiologists, medical physicists, and other medical professionals with more than 53,400 members worldwide. RSNA hosts the world’s premier radiology forum and publishes two top peer-reviewed journals: Radiology, the highest-impact scientific journal in the field, and RadioGraphics, the only journal dedicated to continuing education in radiology.

The Society of Thoracic Radiology (STR) was founded in 1982. The STR is dedicated to advancing cardiothoracic imaging in clinical application, education, and research in radiology and allied disciplines. Continuing professional development opportunities provided by the STR include educational and scientific meetings, mentorship programs, grant support and award opportunities, our society journal, Journal of Thoracic Imaging, and global collaboration activities.

*****Evaluation***

Every study / exam has a row for each label that is scored. It is uniquely indicated by the StudyInstanceUID. Every image, further, has a row for the PE Present on Image label and is uniquely indicated by the SOPInstanceUID. Your prediction file should have a number of rows equal to: (number of images) + (number of studies * number of scored labels).

**Metric

The metric used in this competition is weighted log loss. It is weighted to account for the relative importance of some labels. There are 9 study-level labels and one image-level label, detailed further on the Data page.

**Exam-level weighted log loss

Let y_ij = 1 if label j was annotated to exam i and y_ij = 0, otherwise. Let p_ij be the predicted probability that y_ij = 1:
i = 1, 2, …, N for N exams in the test set
j = 1, 2, …, 9 labels

Let w_j signify the weight for label j.

The weights are as follows:
Label 	Weight
Negative for PE 	0.0736196319
Indeterminate 	0.09202453988
Chronic 	0.1042944785
Acute & Chronic 	0.1042944785
Central PE 	0.1877300613
Left PE 	0.06257668712
Right PE 	0.06257668712
RV/LV Ratio >= 1 	0.2346625767
RV/LV Ratio < 1 	0.0782208589

Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

The binary weighted log loss function for label j on exam i is specified as:
Lij=−wj∗[yij∗log(pij)+(1−yij)∗log(1−pij)]

**image-level weighted log loss

Let y_ik = 1 if image k in exam i was annotated as ‘PE Present on Image’; otherwise, y_ik = 0.
Let p_ik be the predicted probability that y_ik = 1.
w = 0.07361963
i = 1, 2, …, N exams
k = 1, 2, …, n_i, where n_i is the number of images in exam i

Then, let m_i = sum_(k = 1 to n_i) y_ik be the number of positive images in exam i such that
q_i = m_i/n_i is the proportion of positive images in exam i

At the image level, we have a binary classification where the image is classified as PE Present on Image or not (image is negative for PE).

The image-level log loss is written as:

The total loss is the average of all image- and exam-level loss, divided by the average of all row (both image- and exam-level) weights. To get the average of all row weights, sum the weights of all images (q_i*w for each image) and all exam-level labels (w_j for each label j in the test set) and divide by the number of rows.
