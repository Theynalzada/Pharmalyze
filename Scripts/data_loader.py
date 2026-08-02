# Importing Dependencies
import requests
import warnings
import logging
import json
import time
import tqdm

# Filtering out potential warnings
warnings.filterwarnings(action="ignore")

# Creating a custom class
class MedDataLoader:
    # Defining instance attributes
    def __init__(self, filepath: str, base_url: str):
        # Instantiating the base URL
        self.base_url=base_url
        
        # Instantiating the filepath
        self.filepath=filepath
        
    # Defining an instance method to send request via an API
    def send_api_request(self, api_params: dict, max_retries=3):
        # Creating a for loop
        for attempt in range(max_retries):
            # Sending request via an API
            response=requests.get(url=self.base_url, params=api_params)
            
            # Creating a condition based on the status code
            if response.status_code==200:
                # Returning the response in JSON
                return response.json()
            elif response.status_code>=500:
                # Sending the log
                logging.warning(msg=f"Server error {response.status_code}. Retrying in 5 seconds... (Attempt {attempt}/{max_retries})")
                
                # Pausing for 5 seconds
                time.sleep(5)
            else:
                # Raising the error
                response.raise_for_status()
        
        # Raising the error
        response.raise_for_status()
        
    # Defining an instance method to fetch Established Pharmacologic Class (EPC)
    def fetch_epc(self, limit: int):
        # Calling the instance method for EPC
        epc_data=self.send_api_request(api_params={"count": "openfda.pharm_class_epc.exact", "limit": limit})
        
        # Extracting the results
        epc_data=epc_data.get("results") or []
        
        # Extracting only the EPC
        epc_data=[epc.get("term") for epc in epc_data]
        
        # Returning the EPC
        return epc_data
    
    # Defining an instance method to fetch medicine data
    def fetch_medicines(self, epc_limit: int, medicine_limit: int):
        # Sending the log
        logging.info(msg="Data loading process has started!")
        
        # Calling the instance method to fetch EPC data
        epc_data=self.fetch_epc(limit=epc_limit)
        
        # Sending the log
        logging.info(msg=f"{len(epc_data)} EPCs have been identified")
        
        # Creating an empty list to store medicine data for each EPC
        medicine_data=[]
        
        # Looping through each EPC
        for epc in tqdm.tqdm(iterable=epc_data, total=len(epc_data)):
            # Trying to fetch medicines data for corresponding EPC
            try:
                # Calling the instance method to send request via an API
                medicines=self.send_api_request(api_params={"search": f'openfda.pharm_class_epc:"{epc}"', "limit": medicine_limit})
            except Exception as e:
                # Sending the log
                logging.error(msg=f"Failed to fetch data for {epc}. Error: {e}")
                
                # Skipping the current EPC
                continue
            
            # Extracting the results
            medicines=medicines.get("results") or []
            
            # Looping through each medicine
            for medicine in medicines:
                # Extracting the openfda
                openfda=medicine.get("openfda") or {}
                
                # Extracting the information regarding metadata
                generic_name=", ".join(openfda.get("generic_name", []))
                brand_name=", ".join(openfda.get("brand_name", []))
                manufacturer=", ".join(openfda.get("manufacturer_name", []))
                product_type=", ".join(openfda.get("product_type", []))
                route=", ".join(openfda.get("route", []))
                
                # Extracting the information regarding medicine usage
                indications=" ".join(medicine.get("indications_and_usage", []))
                dosage=" ".join(medicine.get("dosage_and_administration", []))
                description=" ".join(medicine.get("description", []))
                
                # Extracting the information regarding safety
                boxed_warning=" ".join(medicine.get("boxed_warning", []))
                drug_warnings=" ".join(medicine.get("warnings_and_cautions", []))
                contraindications=" ".join(medicine.get("contraindications", []))
                drug_interactions=" ".join(medicine.get("drug_interactions", []))
                adverse_reactions=" ".join(medicine.get("adverse_reactions", []))
                overdosage=" ".join(medicine.get("overdosage", []))
                
                # Extracting the information regarding details & pharmacology
                use_in_specific_populations=" ".join(medicine.get("use_in_specific_populations", []))
                mechanism_of_action=" ".join(medicine.get("mechanism_of_action", []))
                clinical_studies=" ".join(medicine.get("clinical_studies", []))
                
                # Extracting the information regarding patients
                information_for_patients=medicine.get("information_for_patients", [])
                spl_medguide=medicine.get("spl_medguide", [])
                patient_info=" ".join(information_for_patients+spl_medguide)
                
                # Grouping non-metadata fields for completeness validation
                non_metadata_fields=[indications,
                                     dosage,
                                     description,
                                     boxed_warning,
                                     drug_warnings,
                                     contraindications,
                                     drug_interactions,
                                     adverse_reactions,
                                     overdosage,
                                     use_in_specific_populations,
                                     mechanism_of_action,
                                     clinical_studies,
                                     patient_info]

                # Creating a condition based on the availablitity of the all non-metadata
                if not all(field.strip() for field in non_metadata_fields):
                    # Skipping the record in a single non-metada is missing
                    continue
                
                # Creating a dictionary
                medicine_dict={"generic_name": generic_name if generic_name else None,
                               "brand_name": brand_name if brand_name else None,
                               "manufacturer": manufacturer if manufacturer else None,
                               "product_type": product_type if product_type else None,
                               "route": route if route else None,
                               "epc": epc,
                               "indications": indications,
                               "dosage": dosage,
                               "description": description,
                               "boxed_warning": boxed_warning,
                               "drug_warnings": drug_warnings,
                               "contraindications": contraindications,
                               "drug_interactions": drug_interactions,
                               "adverse_reactions": adverse_reactions,
                               "overdosage": overdosage,
                               "use_in_specific_populations": use_in_specific_populations,
                               "mechanism_of_action": mechanism_of_action,
                               "clinical_studies": clinical_studies,
                               "information_for_patients": patient_info}
            
                # Updating the dictionary
                medicine_data.append(medicine_dict)
                
                # Sending the log
                logging.info(msg=f"Data for {brand_name} medicine has been loaded!")
                
            # Sending the log
            logging.info(msg=f"Data for {epc} has been loaded!")
        
        # Saving the data
        with open(file=self.filepath, mode="w") as json_data:
            json.dump(obj=medicine_data, fp=json_data, indent=4)
            
        # Sending the log
        logging.info(msg=f"Data loading process finished. Total number of medicines: {len(medicine_data)}")
        
        # Returning the data
        return medicine_data