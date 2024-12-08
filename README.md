# Assignment-4.1
 
#Crawled Fullerton domain
##catalog name is just the starting point of the crawl
##new_catalog is processed, removing some urls that probably dont work

https://1drv.ms/f/c/47f1a052df4b660d/Ep9bkh037vtCs_ulCpeBBc0BIjIoWyUZg9cwXQ2rT8xkVQ?e=4Wmzwm

in the chat loop funciton, on first run, change 

vector_store = load_vector_store()
tokenizer, model = load_llama_model()

to
vector_store = create_vector_store()
tokenizer, model = load_llama_model()

make sure all packages are installed,
i forgot which ones are needed. just install the onese at the top, and run it a couple of times for other dependencies it says you need.

#This is using huggingface, so go the the llama, choose model, and request access. will grant access in several minutes.
https://huggingface.co/meta-llama/Llama-3.2-3B

after access, you need to create an access token from huggingface, then set that token in your environment. 
