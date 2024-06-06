import "dotenv/config";
// a langchain document is an object in a specific format that all langchain vector stores are expecting
import { Document } from "langchain/document";
//in memory vector store (use vector-based db in prod for persistence)
import { MemoryVectorStore } from "langchain/vectorstores/memory";
// creates the embeddings using OpenAI - endpoint where you send data and it returns embeddings
import { OpenAIEmbeddings } from "@langchain/openai";
import { movies } from "./movies.js";

// function that creates the vector store based on the index of movies
// convert movies into langchain document
// see also https://js.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
const createStore = () =>
  MemoryVectorStore.fromDocuments(
    movies.map((movie) => {
      const { title, description, id } = movie;
      return new Document({
        // pageContent is instructure content I want to turn into a vector
        pageContent: `Title: ${title}\n${description}`,
        // structure content you can use after returning the result of the query (e.g. link to website)
        metadata: { source: id, title },
      });
    }),
    // embeddings fn we want the vector store to use to convert all these documents into vectors/embeddings
    // api call
    new OpenAIEmbeddings()
  );

const search = async (query, count = 1) => {
  // count is for how many at most result we want back
  const store = await createStore();
  // convert the query into a vector, throw it into the vector store and get the similarity search
  // doing math to see how close this query (embeddings version) is to other embeddings in the store - cosine similarity
  return store.similaritySearch(query, count);
  //   return store.similaritySearchWithScore(query, count);
};

console.log(await search("something cute and fluffy"));
