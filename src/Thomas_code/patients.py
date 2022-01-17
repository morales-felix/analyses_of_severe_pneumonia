import os
import pandas as pd

from Thomas_code import inout


def modified_edw_rc(table, revision='latest', columns=None):
    """
    Loads modified (e.g.: partially cleaned) tables from
    the merged EDW-RedCap database, sqlsvr01-script-test.database.
    
    Input:
    table     str name of table (see below)
    revision  str name of release, e.g.: '2021-08-31/210831_digest'
    columns   list name of columns to load (recommended for large tables if
                columns of interest are known -> thus to speed up loading)
                
    Output:
    table     pandas dataframe
    


    Tables are:
    patient     contains alternate IDs of patients

    bal_results
    basic_endpoints
    clinical_metadata          contains most clinically relevant infomation
    medication_administration
    organism_codes
    patient
    patient_comorbidities
    sofa_scores
    
    metadata.last_refresh_date
    
    condition_occurrence
    drug_exposure
    measurement
    observation
    observation_period
    person
    procedure_occurrence
    provider
    visit_occurrence
    
    bal_sample
    blood_sample_type
    bronchial_brushing_sample
    demographics
    end_of_study
    
    patient_identifiers
    pneumonia_episode_category_assessment
    pneumonia_episode_outcome
    antibiotic
    bacterial_wgs_metadata
    pipeline_generated_data
    bronchial_brushing
    scscript
    techcore
    """
   
    
    if revision=='latest':
        revision = '2021-12-11/211211_digest_skip_bal_results'
        
    revision = r'{}'.format(revision)

    p_in = inout.get_path(
        'modified_limited', 
        r'sqlsvr01-script-test.database\{}'.format(revision))
    p = os.path.join(p_in, table +'.parquet')
    
    df = pd.read_parquet(p, columns=columns)
        
    return df