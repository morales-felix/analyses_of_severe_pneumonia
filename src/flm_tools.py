import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from statsmodels.sandbox.stats.multicomp import multipletests
import random

from Thomas_code import patients, feature_conversion


def get_cleaned_data(version='latest', outcome_encoding='multiple', multiple_visits=None, remove_unknown_measure=True, restrict=True):
    """
    Returns a cleaned table containing outcomes, measurements, patient age,
    dates, lengths of stay, and patient biological sex
    
    Parameters:
    version: From which extracted database version to pull from. Default is 'latest'.
    Check modeling core's fsmresfile drive
    (as of December 2021, R:\Limited_Data\modelling_core\modified_data\limited\sqlsvr01-script-test.database)
    for possible versions.
    If not latest, string will need to have "yyyy-mm-dd/yymmdd_digest" format.
    
    outcome_encoding: Specify whether discharge disposition categories be encoded in a:
    'binary': 0 for expired or patients discharged to hospice, 1 for other discharges
    'three': 0 just like binary, 1 for discharges to other healthcare facilities, 2 for sent home discharges
    'multiple': Discrete 0 to 5 discharge favorability encoding scheme. This is the default.
    
    multiple_visits: How to deal with patients hospitalized more than once.
    None: Default option: Discard patients with multiple encounters. 
    'first': Keep the first encounter/hospitalization, discard the rest.
    'last': Opposite to first, keep last encounter, discard the rest.
    'all': Keep all encounters.
    
    remove_unknown_measure: Some clinical measurements are named "No matching concept".
    Default is True, and it eliminates unnamed measurements.
    
    restrict: Whether to filter out measurements not administered during visit/encounter.
    Default is True.
    
    """
    
    # Outcomes
    df = patients.modified_edw_rc('basic_endpoints', revision=version)
    df = feature_conversion.add_discharge_disposition_name_as_number(df, scheme=outcome_encoding)
    outcomes = df[['case_number', 'discharge_disposition_name_conv']].dropna(
        subset=['discharge_disposition_name_conv']).astype(int).drop_duplicates()   # Slicing table to just get the patient identifier and a distinct outcome
    outcomes = outcomes.drop_duplicates(subset=['case_number'], keep=False)  # Eliminating any case appearing with more than one outcome
    if outcomes['case_number'].value_counts().max() > 1:
        raise AssertionError('Case number tied to multiple discharge disposition categories')
        
    
    # Measurements. First, defining which columns to read in, which can reduce function runtime
    columns_to_keep = ['cohort_patient_id', 'measurement_vocabulary_id', 'measurement_concept_code',
                       'measurement_concept_name', 'measurement_concept_class_id', 'measurement_datetime',
                       'measurement_type_concept_name', 'operator_concept_id', 'operator_concept_name',
                       'operator_vocabulary_id', 'operator_concept_class_id', 'operator_concept_code',
                       'value_as_number', 'value_as_concept_id', 'value_as_concept_name',
                       'value_as_vocabulary_id', 'value_as_concept_class_id', 'value_as_concept_code',
                       'unit_concept_id', 'unit_concept_name', 'unit_vocabulary_id',
                       'unit_concept_class_id', 'unit_concept_code']
    full_measurement_table = patients.modified_edw_rc('measurement',
                                                      revision=version,
                                                      columns=columns_to_keep)
    
    if remove_unknown_measure:
        g = full_measurement_table['measurement_concept_name'] == 'No matching concept'
        full_measurement_table.loc[g, 'measurement_concept_name'] = np.nan
        full_measurement_table = full_measurement_table.dropna(subset=['measurement_concept_name'])
        
        
    # Mapper between different identifiers used in these tables
    map_patient_ids = patients.modified_edw_rc('patient', revision=version)
        
        
    # Patient age
    patient_identifiers = patients.modified_edw_rc(
        'patient_identifiers', revision=version, columns=['case_number', 'pt_age']).dropna()
    
    
    # Dates and lengths of stay (LOS)
    if multiple_visits == 'first':
        earliest = df.groupby('patient_ir_id')['admission_datetime'].min().to_frame('admission_datetime').reset_index()
        dates_and_LOS = pd.merge(df[['patient_ir_id', 'case_number', 'admission_datetime', 'discharge_datetime', 'hospital_los_days']],
                                 earliest,
                                 how='inner').dropna()
    elif multiple_visits == 'last':
        latest = df.groupby('patient_ir_id')['admission_datetime'].max().to_frame('admission_datetime').reset_index()
        dates_and_LOS = pd.merge(df[['patient_ir_id', 'case_number', 'admission_datetime', 'discharge_datetime', 'hospital_los_days']],
                                 latest,
                                 how='inner').dropna()
        
    elif multiple_visits == 'all':
        dates_and_LOS = df[['patient_ir_id', 'case_number', 'admission_datetime', 'discharge_datetime', 'hospital_los_days']].dropna()
        
    elif multiple_visits is None:
        dates_and_LOS = df[['patient_ir_id', 'case_number', 'admission_datetime', 'discharge_datetime', 'hospital_los_days']].dropna()
        map_patient_ids = map_patient_ids.drop_duplicates('cohort_patient_id', keep=False)   # Only keep patients with one encounter
        if map_patient_ids['cohort_patient_id'].value_counts().max() > 1:
            raise AssertionError('Multiple cohort_patient_id')        
    else:
        raise TypeError("Did not specify one of three ways to deal with multiple hospitalizations ('first', 'last', 'all', or None)")
        
    
    # Biological sex
    person = patients.modified_edw_rc('person', revision=version, columns=['cohort_patient_id',
                                                                           'gender_concept_name',
                                                                           'race_concept_name',
                                                                           'ethnicity_concept_name'])
    
    
    
    # Joining all tables.
    clean_all_table = pd.merge(map_patient_ids, full_measurement_table)
    clean_all_table = pd.merge(clean_all_table, outcomes)
    clean_all_table = pd.merge(clean_all_table, person)
    clean_all_table = pd.merge(clean_all_table, dates_and_LOS)
    clean_all_table = pd.merge(clean_all_table, patient_identifiers)
    
    
    # Converting data types
    try:
        clean_all_table['pt_age'] = clean_all_table['pt_age'].astype(int)
    except TypeError:
        print("Failure to convert pt_age to int")
    
    try:
        clean_all_table['gender_concept_name'] = clean_all_table['gender_concept_name'].astype(str).str.capitalize()
    except TypeError:
        print("Failure to convert gender to str. Likely, at least one patient has gender missing")
    
    
    if restrict:
        clean_all_table = clean_all_table.loc[
            (clean_all_table['measurement_datetime'] >= clean_all_table['admission_datetime']) &
            (clean_all_table['measurement_datetime'] <= clean_all_table['discharge_datetime'])]
        
        
    return clean_all_table



def add_column_with_day_of_stay(table):
    """
    Input would be the clean_all_table (output of get_cleaned_data).
    Adds a column with the day of stay of each patient, and also adds
    another column with the measurement order.
    Day of stay is taken to be day 1 on the day of admission,
    regardless of time of admission.
    
    
    table must be a pandas DataFrame
    """
    
    cases = list(table['case_number'].drop_duplicates())
    frames = []
    
    for case in cases:
        f = table['case_number'] == case
        df = table.loc[f]
        df = df.sort_values('measurement_datetime')
        df['day_of_visit'] = (df['measurement_datetime'].dt.date - df['admission_datetime'].dt.date).dt.days + 1
        df['order_of_measurement'] = df['measurement_datetime'].rank(method='dense', na_option='bottom').astype(int)
        frames.append(df)
        
    clean_all_table = pd.concat(frames)
    
    clean_all_table = clean_all_table.sort_values(by=['case_number',
                                                      'measurement_datetime'])
    
    return clean_all_table


def parse_reference_normal(row, low=True):
    """If reference range cutoffs are in reference_normal column,
    but not in reference_low or reference_high,
    this function will parse the values and fill in the empty entries.
    IMPORTANT: You gotta mask/subset the pandas df beforehand.
    This function is meant to be used through the .apply() pandas method.
    It is very specific to a preliminary analysis use case,
    but it could be extended if useful.
    
    
    Input:
    row: a specific row in the dataframe.
    low: whether you are trying to fill the reference_low column or not
         this will determine which column to focus on.
         
    Return:
    float - the numeric value of the relevant reference range cutoff
            If low=True, return lower bound -- Default
            If low=False, return upper bound
    """
        
    if '-' in row['reference_normal']:
        
        split_range = row['reference_normal'].split("-")
        
        try:
            # Trying to convert elements to float so we can catch the case where "based on documented sex..." occurs
            split_range = [float(i) for i in split_range]
        except ValueError:
            split_range = [split_range[0].split(" ")[-1], split_range[-1]]
            split_range = [float(i) for i in split_range]
                
    elif '<=' in row['reference_normal']:
        # One case where reference normal is "<= 0.065". We assume the lower bound is 0.0.
        split_range = row['reference_normal'].split("<=")
        split_range[0] = '0.0'
        split_range = [float(i) for i in split_range]
        
    else:
        split_range = [None, None]
            
    if low:
        # Returning as string so conversion to numeric type is done afterwards (and to not mix up data types in the column)
        return str(split_range[0])
    else:
        return str(split_range[1])
    
    
def get_cdfs(dist_data):
    """
    Returns the sorted unique value of dist_data, and a list of percentiles (after normalizing)
    """
    
    data_sorted = np.sort(dist_data)

    perc = np.arange(len(dist_data)) / (len(dist_data) - 1)
    
    return data_sorted,perc


def get_survival_fn(dist_data):
    """
    Returns the sorted unique value of dist_data, and a backwards list of percentiles (after normalizing)
    
    log_scale: Whether to report dist_data values in a log scale
    """
    
    data_sorted = np.sort(dist_data)

    perc = ((len(dist_data) - 1) - np.arange(len(dist_data))) / (len(dist_data) - 1)
    
    return data_sorted,perc


def confidence_interval(df, ci_width=95):
    """
    This function returns confidence intervals using Student's t-distribution method
    (a.k.a. NOT bootstrapped).
    It will specifically return the upper and lower bounds of the CI as separate
    pandas dataframe columns.
    
    Specify the CI through ci_width; only two options: 95 (default) and 99.
    """
    
    ci_hi = []
    ci_lo = []
    
    if ci_width == 95:
        for i in df.index:
            m, c, s = df.loc[i]
            ci_hi.append(m + 1.95996*s/math.sqrt(c))
            ci_lo.append(m - 1.95996*s/math.sqrt(c))
            
    elif ci_width == 99:
        for i in df.index:
            m, c, s = df.loc[i]
            ci95_hi.append(m + 2.57583*s/math.sqrt(c))
            ci95_lo.append(m - 2.57583*s/math.sqrt(c))

    df['ci_hi'] = ci_hi
    df['ci_lo'] = ci_lo
    
    return df.reset_index()


    
### Functions by Thomas Stoeger ###
def _add_bonferroni(counts):
    counts = counts.copy()
    counts.loc[:, 'bonferroni'] = multipletests(
        counts.loc[:, 'pvalue'].values, method='bonferroni')[1]

    f = counts['bonferroni'] == 0
    counts.loc[~f, 'log_bonferroni'] =  -np.log10(counts.loc[~f, 'bonferroni'])
    counts.loc[f, 'log_bonferroni'] = np.inf
    return counts


# By Thomas Stoeger, modified by Felix Morales to suit other needs
def enrichment(entity2category, entity2annotation, entity, category, annotation, category_in_annotation,
               category_not_in_annotation, only_entities_with_annotation=True, drop_duplicates=True):
    """
    Computes enrichments.

    Input:
        entity2category    df with entities (e.g.: pubmed_id) and category (e.g. 'new') that must be boolean
        entity2annotation  df with entities (e.g.: pubmed_id) and annotation (e.g.: mesh_term)
        entity             str 
        category           str
        annotation         str
        only_entities_with_annotation       default True, will only consider entities with at least one annotation
        drop_duplicates    default True; ensures that each pairing of entity to annotation and category only occurs once
    """

    if entity2category[entity].value_counts().max() > 1:
        print('entity2category is not unique')

    if not all(entity2category[category].isin([False, True])):
        raise AssertionError(
            'category must have either True or False as values.')

    entity2annotation = entity2annotation.loc[:, [entity, annotation]].copy()
    entity2category = entity2category.loc[:, [entity, category]].copy()

    entity2annotation = entity2annotation[
        entity2annotation[entity].isin(entity2annotation[entity])]

    if only_entities_with_annotation:
        entity2category = entity2category[entity2category[entity].isin(
            entity2annotation[entity])]

    if drop_duplicates:
        entity2annotation = entity2annotation.drop_duplicates()
        entity2category = entity2category.drop_duplicates()

    observed = entity2category[category].value_counts().reindex([False, True]).fillna(0)
    total_hits = observed[True]
    total_non_hits = observed[False]

    toy = pd.merge(entity2annotation, entity2category)

    counts = toy.groupby(
        [annotation, category]
    ).size().to_frame('v').reset_index().pivot(
        index=annotation, columns=category, values='v').fillna(0)

    counts = counts.reindex(columns=[True, False]).fillna(0)

    counts = counts.rename(columns={
        True: category_in_annotation[0],
        False: category_in_annotation[1]
    })

    counts.loc[:, category_not_in_annotation[0]] = total_hits - \
        counts.loc[:, category_in_annotation[0]]
    counts.loc[:, category_not_in_annotation[1]] = total_non_hits - \
        counts.loc[:, category_in_annotation[1]]

    for term, row in counts.iterrows():

        odds_ratio, pvalue = fisher_exact(
            [
                [row[category_in_annotation[0]], row[category_not_in_annotation[0]]],
                [row[category_in_annotation[1]], row[category_not_in_annotation[1]]]
            ],
            alternative='greater'
        )
        counts.loc[term, 'odds_ratio'] = odds_ratio
        counts.loc[term, 'pvalue'] = pvalue

    for term, row in counts.iterrows():

        odds_ratio, pvalue = fisher_exact(
            [
                [row[category_in_annotation[0]], row[category_in_annotation[1]]],
                [row[category_not_in_annotation[0]], row[category_not_in_annotation[1]]]
            ],
            alternative='two-sided'  # 'greater'
        )
        counts.loc[term, 'odds_ratio'] = odds_ratio
        counts.loc[term, 'pvalue'] = pvalue

    counts['fraction_survived'] = counts[category_in_annotation[0]] / total_hits
    counts['fraction_dead'] = counts[category_in_annotation[1]] / total_non_hits
    counts['fraction_all'] = (
        counts[category_in_annotation[0]] + counts[category_in_annotation[1]]) / (total_hits + total_non_hits)

    # note that fraction_all must be >0 given above code
    f = (counts['fraction_survived'] > 0)
    counts.loc[f, 'ratio'] = np.log2(
        counts.loc[f, 'fraction_survived'] / counts.loc[f, 'fraction_all'])
    counts.loc[~f, 'ratio'] = -np.inf  # would be used for log2(0)

    counts = _add_bonferroni(counts)
    counts = counts.reset_index()
    
    return counts


# By Thomas Stoeger, modified by Felix Morales to suit other needs
def plot_step(enrichment_results, category_in_annotation, graph_title, sig_thresh = 0.01, namer=None):
    """
    Will make a scatter plot for enrichment results, if namer is
    not None, significant dots will be labeled according
    to the column name defined by namer

    """

    a = enrichment_results.copy()

    f = a['log_bonferroni'] >= -np.log10(sig_thresh)
    f_increase = a['ratio'] > 0

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7,7))

    ax.scatter(np.log10(a.loc[~f, category_in_annotation[1]] + 1),
               np.log10(a.loc[~f, category_in_annotation[0]] + 1),
               c='darkgray')
    ax.scatter(np.log10(a.loc[f & f_increase, category_in_annotation[1]] + 1),
               np.log10(a.loc[f & f_increase, category_in_annotation[0]] + 1),
               c='blue')
    ax.scatter(np.log10(a.loc[f & ~f_increase, category_in_annotation[1]] + 1),
               np.log10(a.loc[f & ~f_increase, category_in_annotation[0]] + 1),
               c='r')

    if namer != None:

        for j in a[f].index:
            ax.text(np.log10(a.loc[j, category_in_annotation[1]]+1)-0.15,
                        np.log10(a.loc[j, category_in_annotation[0]]+1)-0.1,
                        a.loc[j, namer],
                        fontsize=15)

    ax.set_xlabel(f"{category_in_annotation[1]}"+ r" ($log_{10}$)", fontsize=20)
    ax.set_ylabel(f"{category_in_annotation[0]}"+ r" ($log_{10}$)", fontsize=20)
    ax.set_title(graph_title, fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(linestyle=':')
    

# By Thomas Stoeger, modified by Felix Morales to suit other needs
def unit_converter(value, source, target):
    """
    Converts value from the source unit_concept_id to
    the target unit_concept_id
    
    Input:
        value   numerical value
        source  unit_concept_id (source)
        target  unit_concept_id (target)
        
    Value:
        new     transformed numerical value
    
    """
     
    new = []
            
    if source == 9639.0:   # Siemens
        if target == 8555.0:  # seconds
            new = value
            
    elif source is np.nan:
        if target == 8848.0:   # thousand per microliter
            new = value
    
    elif source == 8840.0:   # milligram per deciliter
        if target == 8751.0:  # milligram per liter
            new = value / 10
             
    else:
        raise AssertionError(
            'Could not find matching source unit_concept_id'
        )
            
    if new == []:
        raise AssertionError(
            'Could not convert source to target. Possibly not implemented.')
        
    return new


# By Thomas Stoeger, modified by Felix Morales to suit other needs
def convert_to_standard_units(measurements):
    """
    Will transform numeric measurements in table
    measurements to standard units.
    
    Input:
    measurements  df with columns 'value_as_number' 
                  and 'unit_concept_id'; further, optionally:
                  'unit_concept_name', and 'unit_concept_code'
    """

    to_transform = [
        {
            'measurement': [
                '14979-9',  # aPTT in Platelet poor plasma by Coagulation assay
                '5902-2'  # Prothrombin time (PT)
            ],
            'from': 9639.0,    # Siemens
            'to': 8555.0       # seconds
        },
        {
            'measurement': [
                '704-7',      # Basophils [#/volume] in Blood by Automated count
                '711-2',      # Eosinophils [#/volume] in Blood by Automated count
                '712-0',      # Eosinophils [#/volume] in Blood by Manual count
                '731-0',      # Lymphocytes [#/volume] in Blood by Automated count
                '742-7',      # Monocytes [#/volume] in Blood by Automated count
                '751-8',      # Neutrophils [#/volume] in Blood by Automated count
                '753-4',      # Neutrophils [#/volume] in Blood by Manual count
                '777-3'      # Erythrocyte distribution width [Ratio] by Automated count
            ],
            'from': np.nan,    # No unit reported
            'to': 8848.0       # thousand per microliter

        },
        {
            'measurement': [
                '16503-5',     # C reactive protein [Mass/volume] in Body fluid
                '1988-5'      # C reactive protein [Mass/volume] in Serum or Plasma
            ],
            'from': 8840.0,    # milligram per deciliter
            'to': 8751.0      # milligram per liter
        }
    ]

    if 'unit_concept_name' in measurements.columns:
        names = measurements[[
            'unit_concept_id',
            'unit_concept_name',
            'unit_concept_code'
        ]].drop_duplicates().dropna().set_index('unit_concept_id')


    for elements in to_transform:
        in_scope_measurement = measurements['measurement_concept_code'].isin(elements['measurement'])
        if elements['from'] is np.nan:
            in_scope_source_unit = in_scope_source_unit = measurements['unit_concept_id'].isna()
        else:
            in_scope_source_unit = measurements['unit_concept_id'] == elements['from']

        f = in_scope_measurement & in_scope_source_unit

        measurements.loc[f, 'value_as_number'] = measurements.loc[f,
                                                              'value_as_number'].apply(lambda x: unit_converter(x,
                                                                                                                elements['from'],
                                                                                                                elements['to'])).copy()

        measurements.loc[f, 'unit_concept_id'] = elements['to']
        measurements.loc[f, 'unit_concept_name'] = names.loc[elements['to'], 'unit_concept_name']
        measurements.loc[f, 'unit_concept_code'] = names.loc[elements['to'], 'unit_concept_code']
            
    return measurements
                
                
                
### Functions by Ritika Giri for logistic regression ###
def logit2prob(log_odds):
    odds = np.exp(log_odds)
    prob = odds/(1+odds)
    return prob


def scatter_text(x, y, text_column, data, title, xlabel, ylabel):
    """Scatter plot with annotation codes on the x y coordinates
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, data=data, size = 20, legend=False)
    # Add text besides each point
    for line in range(0,data.shape[0]):
         p1.text(data[x][line]+0.001, data[y][line], 
                 data[text_column][line], horizontalalignment='left', 
                 size='xx-small', color='red', weight='ultralight')
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1


def results_summary_to_dataframe(results):
    """
    Create pandas dataframe from statsmodels regression result
    """
    #model = sm.OLS(y,x)
    #results = model.fit()
    
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int(alpha=0.01)[0]
    conf_higher = results.conf_int(alpha=0.01)[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df


def logistic_reg_outputter(model, title):
    
    # ------ print important outputs --------#
    print('\n\nLikelihood Ratio : ',model.llr)
    print('LR p-value : ', model.llr_pvalue)
    display(model.summary().tables[0])
    
    print('Prediction table\n', model.pred_table())
    specificity = model.pred_table()[0,0]/(model.pred_table()[0,0] + model.pred_table()[0,1])
    sensitivity = model.pred_table()[1,1]/(model.pred_table()[1,1] + model.pred_table()[1,0])

    print('Specificity: ', specificity)
    print('Sensitivity: ', sensitivity)
    
    # ------ format result table into dataframe --------#
    table = results_summary_to_dataframe(model)
    table['prob'] = table['coeff'].apply(logit2prob)
    table = table.sort_values(by=['prob', 'pvals', 'coeff'], ascending=False)
    table['input'] = table.index
    table['input'] = table.input.apply(lambda x: x.split('[')[-1])
#     table['input'] = table.input.apply(lambda x: x[:-1])
    display(table)
    
    # ------ scatter plot of regression coefficients -------#
#     scatter_text(x='prob', y='pvals', text_column='input', data=table, 
#                  title = title, 
#                  xlabel='Estimated Probability from Coefficient', 
#                  ylabel='p value of coefficient')
# #     plt.xlim([0,1]) # probability limits
#     plt.yscale('log')
#     plt.show()
    # ------ residual index plot ---------------#
    resid = pd.DataFrame(model.resid_pearson)
    resid = resid.reset_index()
    resid.columns = ['Index', 'Pearson Residual']
    
    sns.scatterplot(x= 'Index',y='Pearson Residual', data=resid)
    plt.title("Residual vs Observation")
    plt.show()
    
    return table


def compare_models(m_simple, m_complex):
    """
    Compare two models using Likelihood ratio and compute the p-value of the test statistic
    
    Args:
        m_simple -- statsmodels logitresult object with fewer (nested) parameters
        m_complex -- statsmodels logitresult object with more parameters
    """
    # get log likelihoods and degrees of freedom
    ll_simple, df_simple = m_simple.llf, m_simple.df_model
    ll_complex, df_complex = m_complex.llf, m_complex.df_model
    # compute Likelihood ratio
    LR = -2*(ll_simple - ll_complex)
    excess_df = df_complex - df_simple
    # compare to chi-square distrubition with df degrees of freedom
    pval = chi2.sf(LR, excess_df)
    print('Likelihood ratio : ', LR)
    print('p-value of Likelihood ratio : ', pval)
    
    return