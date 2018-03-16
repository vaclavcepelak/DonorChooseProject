# Donors Choose Application screening

This repository contains the codes for the [DonorsChoose project](https://www.kaggle.com/c/donorschoose-application-screening/) - a Kaggle competition.

Founded in 2000 by a high school teacher in the Bronx, [DonorsChoose.org](www.donorschoose.org) empowers public school teachers from across the country to request much-needed materials and experiences for their students. At any given time, there are thousands of classroom requests that can be brought to life with a gift of any amount.

The goal of the project is to predict whether the application will or will not approved, i.e. a binary classification problem. The applications data are mostly textual data. The codes are fully reproducible.

## File descriptions
`train.csv` - the training set
`test.csv` - the test set
`resources.csv` - resources requested by each proposal; joins with test.csv and train.csv on id
`sample_submission.csv` - a sample submission file in the correct format
Data fields

`test.csv` and `train.csv`:

`id` - unique id of the project application
`teacher_id` - id of the teacher submitting the application
`teacher_prefix` - title of the teacher's name (Ms., Mr., etc.)
`school_state` - US state of the teacher's school
`project_submitted_datetime` - application submission timestamp
`project_grade_category` - school grade levels (PreK-2, 3-5, 6-8, and 9-12)
`project_subject_categories` - category of the project (e.g., "Music & The Arts")
`project_subject_subcategories` - sub-category of the project (e.g., "Visual Arts")
`project_title` - title of the project
`project_essay_1` - first essay*
`project_essay_2` - second essay*
`project_essay_3` - third essay*
`project_essay_4` - fourth essay*
`project_resource_summary` - summary of the resources needed for the project
`teacher_number_of_previously_posted_projects` - number of previously posted applications by the submitting teacher
`project_is_approved` - whether DonorsChoose proposal was accepted (`0` = "rejected", `1` = "accepted"); `train.csv` only

`resources.csv`:

Proposals also include resources requested. Each project may include multiple requested resources. Each row in resources.csv corresponds to a resource, so multiple rows may tie to the same project by id.

`id` - unique id of the project application; joins with test.csv. and train.csv on id
`description` - description of the resource requested
`quantity` - quantity of resource requested
`price` - price of resource requested

## The modelling methods

For the memory reasons, **only 30,000 cases** from the train set were used for modelling.

The **gradient boosting machine** is applied for the binary classification problem. As the features are mostly textual data, the **topic models** (**Latent Dirichlet Allocation**) are applied on essays and on resources to generate numeric features from the textual data. Features are computed as a probability of a text being assigned under a certain topic (75 topics for essays, 20 topics for resources). All the variables with multiple assignments per one application are aggregated using several functions (`mean`, `sd`, `min`, `max`, `.N`).

The `gbm` library (with the`caret` library as a wrapper) in R is used; 4-fold cross validation is performed and grid search is applied for the parameter tuning.

The `data.table` library is mostly used for data transformation (for speed reasons).

## Scripts 

All scripts are stored in the `/scripts` folder. All the outputs of each script are stored to an the `/output` folder. 

`data_train.R` : Performs data cleaning, transformation and feature engineering for the train data (saves the data and the topic models to the `.rda` files)

`modelling.R` : Performs the feature selection and gbm modelling using grid search; gbm model exported to an `.rda` file + additional outputs (ROC curve, confusion matrix) are exported

`data_test.R` : The data cleaning is applied on the test sample and topic scores are predicted based on the models from the train set (data exported to an `.rda` file)

`prediction.R` : A simple script which takes the test data and gbm model and predicts the scores for submission (exports to csv)

`helpers.R` : Contains helper functions for data transformation and feature engineering