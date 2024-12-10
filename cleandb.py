from pymongo import MongoClient, UpdateOne
import re
from config import *

#Connect to MongoDB
client = MongoClient(MONGO_HOST, MONGO_PORT)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]
new_collection = db[NEW_MONGO_COLLECTION]

def copy_collection():
    #Copies all Documents from one collection to another

    if new_collection.count_documents({}) > 0:
        print(f"New collection '{NEW_MONGO_COLLECTION}' already exists and is not empty. Skipping copy.")
        return
    
    print(f"Copying documents from '{MONGO_COLLECTION}' to '{NEW_MONGO_COLLECTION}'...")
    documents = list(collection.find())
    if documents:
        new_collection.insert_many(documents)
    print(f"Copied {len(documents)} documents to '{NEW_MONGO_COLLECTION}'.")


def normalize_url(url):
    #normalize the url, removing duplicates
    # for catalog.fullertion, anything after preview_entity's returnto=xx seems to me unncessary. 
    pattern = r"(https:\/\/catalog\.fullerton\.edu\/preview_entity\.php\?[^&]+&[^&]+&returnto=\d+)"
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    return url  # Return the original URL if it doesn't match the pattern


def clean_urls_in_collection():
    #Cleans the URL for all documents in the new collection
    processed_count = 0
    bulk_updates = []

    print(f"Cleaning URLs in '{NEW_MONGO_COLLECTION}, {new_collection.count_documents({})}'...")

    #Only ID and URL are needed for this
    cursor = new_collection.find({}, {"_id": 1, "url": 1})

    for doc in cursor:
        original_url = doc.get("url", "")
        if not original_url:
            continue

        cleaned_url = normalize_url(original_url)
        
        #If the url got Cleaned
        if cleaned_url != original_url:
            print(f"Cleaning URL: {original_url} -> {cleaned_url}")
            bulk_updates.append(
                UpdateOne(
                    {"_id": doc["_id"]},  # Make sure _id is referenced correctly
                    {"$set": {"url": cleaned_url}}
                )
            )
            processed_count += 1

    if bulk_updates:
        try:
            result = new_collection.bulk_write(bulk_updates)
            print(f"Updated {result.modified_count} documents with cleaned URLs.")
        except Exception as e:
            print(f"Bulk write failed: {e}")

    print(f"Processed {processed_count} documents and cleaned URLs.")

def remove_unwanted_domains():
    #Remove any Document that is part of these Domains
    unwanted_domains = [
        "titansgive.fullerton",
        "giveto.fullerton",
        "give.fullerton"
    ]
    delete_query = {
        "url": {
            "$regex": f"({'|'.join(unwanted_domains)})"
        }
    }
    try:
        delete_result = new_collection.delete_many(delete_query)
        print(f"Deleted {delete_result.deleted_count} documents with unwanted URLs from '{NEW_MONGO_COLLECTION}'.")
    except Exception as e:
        print(f"Error during delete_many: {e}")

def count_domains():
    #Counts the frequency of each Domain 

    print(f"Counting domain occurrences in '{NEW_MONGO_COLLECTION}'...")

    # Extract the domain from the URL using regex and count occurrences
    cursor = new_collection.find({}, {"url": 1})
    domain_counts = {}

    for doc in cursor:
        url = doc.get("url", "")
        if not url:
            continue

        # Extract domain using regex (e.g., catalog.fullerton.edu, titansgive.fullerton.edu)
        match = re.match(r"https?://([^/]+)", url)
        if match:
            domain = match.group(1)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nDomain counts (most common first):")
    for domain, count in sorted_domains:
        print(f"{domain}: {count} occurrences")

    return sorted_domains

def remove_unwanted_urls():
    #Remove Unwanted URLS.
    delete_query = {
        "url": {
            "$regex": r"(content\.php|index\.php)"
        }
    }
    
    try:
        delete_result = new_collection.delete_many(delete_query)
        print(f"Deleted {delete_result.deleted_count} documents with unwanted URLs from '{MongoClient}'.")
    except Exception as e:
        print(f"Error during delete_many: {e}")


non_matching_regex = r"https:\/\/catalog\.fullerton\.edu\/(?!content\.php|preview_program\.php)"
def count_non_matchin_urls():
    #This was used for Testing to see how many URLS didnt meet this pattern
    query = {
        "url": {
            "$regex": non_matching_regex
        }
    }
    
    try:
        count = new_collection.count_documents(query)
        print(f"Found {count} documents with non-matching URLs in '{NEW_MONGO_COLLECTION}'.")
    except Exception as e:
        print(f"Error during count_documents: {e}")
def remove_non_matching_urls():
    #Removed any url part of the catalog.fullerton domain that first subdirectory does not have
    #content.php or preview_program.php
    query = {
        "url": {
            "$regex": non_matching_regex
        }
    }
    
    try:
        delete_result = new_collection.delete_many(query)
        print(f"Deleted {delete_result.deleted_count} documents with non-matching URLs from '{NEW_MONGO_COLLECTION}'.")
    except Exception as e:
        print(f"Error during delete_many: {e}")

if __name__ == "__main__":

    print("\nCounting domains")
    #count_domains()

    print("Copy collection to new collection...")
    #copy_collection()
    
    print("\nCleaning URLs in new collection...")
    clean_urls_in_collection()
    
    print("\nRemoving documents with unnecessary URLs...")
    remove_unwanted_urls()

    print("\nCount Old urls")
    #count_non_matchin_urls()

    print("\nRemove Old urls")
    remove_non_matching_urls()

    print("\nRemove unwanted domains")
    #remove_unwanted_domains()

    print("\nCounting domains")
    #count_domains()

