import numpy as np

def shorten_pt_category(df):
    """
    Replaces detailed desciption within pt_category by
    
    shorter description via 1:1.
    """

    df['pt_category'] = df['pt_category'].replace({
        'Clinical HAP (current admission >48 hours or discharged from a healthcare facility within the last 7 days where admission >24 hours)': 'HAP',
        'Clinical CAP (not hospitalized within the last 7 days)': 'CAP',
        'Clinical VAP (on ventilator >48 hours or reintubated < 24 hours from extubation)': 'VAP',
        'Non-pneumonia control': 'non'
        }
    )
    return df
       
    
def add_pt_educ_level_as_number(df):
    """
    Adds pt_educ_level desciption as number, going
    from lowest to highest level and stores it
    in column pt_educ_level_conv
    """
    
    # ranking as in https://github.com/NUSCRIPT/script_etl_eda/issues/60
    df['pt_educ_level_conv'] = df['pt_educ_level'].copy().replace({
        'Less than high school':1,
        'High school graduate':2,
        'Some college': 3, 
        'College graduate': 4,
        'Advanced degree': 5
        }
    )
    return df
    
    
def add_discharge_disposition_name_as_number(df, scheme='multiple'):
    """
    Adds discharge_disposition_name from least favorable to most favorable
    and stores it in column discharge_disposition_name_conv.
    If scheme='multiple', discharges encoded as follows:

    'Home or Self Care': 5,
    'Group Home': 5,
    'Home with Equipment or O2': 5,
    'Home with Outpatient Services': 5,
    'Home with Home Health Care': 5,
    'Against Medical Advice (AMA) or Elopement': 5,
    'Inpatient Psychiatric Hospital': 5,
    'Acute Inpatient Rehabilitation': 4,
    'Planned Readmission – DC/transferred to acute inpatient rehab': 4,
    'Skilled Nursing Facility or Subacute Rehab Care': 3,
    'Nursing Home (Custodial)': 3,
    'Long-Term Acute Care Hospital (LTACH)': 2,
    'Long-Term Acute Care Hospital (LTAC)': 2,
    'Acute Care Hospital': 2,
    'Home with Hospice': 1,
    'Inpatient Hospice': 1,
    'Expired': 0,
    'unknown': np.nan
    
    If scheme='binary', then discharge dispositions are categorized as:
    Death: 0 (Expired, Inpatient Hospice, Home with Hospice)
    Survival: 1 (All other discharge categories)
    
    If scheme='three', then:
    Death: 0 (Expired, Inpatient Hospice, Home with Hospice)
    Healthcare Transfer: 1 (LTAC, Nursing Home, Acute Inpatient Rehabilitation)
    Home: 2 (Home, Against Medical Advice, Inpatient Psychiatric Hospital)
   
    """
    
    if scheme=='multiple':
        # ranking as in latest version of https://github.com/NUSCRIPT/script_etl_eda/issues/21
        df['discharge_disposition_name_conv'] = df['discharge_disposition_name'].copy().replace({
            'Home or Self Care': 5,
            'Group Home': 5,
            'Home with Equipment or O2': 5,
            'Home with Outpatient Services': 5,
            'Home with Home Health Care': 5,
            'Against Medical Advice (AMA) or Elopement': 5,
            'Inpatient Psychiatric Hospital': 5,
            'Acute Inpatient Rehabilitation': 4,
            'Planned Readmission – DC/transferred to acute inpatient rehab': 4,
            'Skilled Nursing Facility or Subacute Rehab Care': 3,
            'Nursing Home (Custodial)': 3,
            'Long-Term Acute Care Hospital (LTACH)': 2,
            'Long-Term Acute Care Hospital (LTAC)': 2,
            'Acute Care Hospital': 2,
            'Home with Hospice': 1,
            'Inpatient Hospice': 1,
            'Expired': 0,
            'unknown': np.nan
            }
        )
        
    elif scheme=='binary':
        df['discharge_disposition_name_conv'] = df['discharge_disposition_name'].copy().replace({
            'Home or Self Care': 1,
            'Group Home': 1,
            'Home with Equipment or O2': 1,
            'Home with Outpatient Services': 1,
            'Home with Home Health Care': 1,
            'Against Medical Advice (AMA) or Elopement': 1,
            'Inpatient Psychiatric Hospital': 1,
            'Acute Inpatient Rehabilitation': 1,
            'Planned Readmission – DC/transferred to acute inpatient rehab': 1,
            'Skilled Nursing Facility or Subacute Rehab Care': 1,
            'Nursing Home (Custodial)': 1,
            'Long-Term Acute Care Hospital (LTACH)': 1,
            'Long-Term Acute Care Hospital (LTAC)': 1,
            'Acute Care Hospital': 1,
            'Home with Hospice': 0,
            'Inpatient Hospice': 0,
            'Expired': 0,
            'unknown': np.nan
            }
        )
        
    elif scheme=='three':
        df['discharge_disposition_name_conv'] = df['discharge_disposition_name'].copy().replace({
            'Home or Self Care': 2,
            'Group Home': 2,
            'Home with Equipment or O2': 2,
            'Home with Outpatient Services': 2,
            'Home with Home Health Care': 2,
            'Against Medical Advice (AMA) or Elopement': 2,
            'Inpatient Psychiatric Hospital': 2,
            'Acute Inpatient Rehabilitation': 1,
            'Planned Readmission – DC/transferred to acute inpatient rehab': 1,
            'Skilled Nursing Facility or Subacute Rehab Care': 1,
            'Nursing Home (Custodial)': 1,
            'Long-Term Acute Care Hospital (LTACH)': 1,
            'Long-Term Acute Care Hospital (LTAC)': 1,
            'Acute Care Hospital': 1,
            'Home with Hospice': 0,
            'Inpatient Hospice': 0,
            'Expired': 0,
            'unknown': np.nan
            }
        )
    return df