import { NextRequest } from "next/server";
import { Index, Pinecone, RecordMetadata } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings, OpenAI } from "@langchain/openai";
import { StreamingTextResponse, LangChainStream } from "ai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
import { VectorOperationsApi } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch/db_data";

export async function POST(request: NextRequest) {
  // Parse the POST request's JSON body
  const body = await request.json();

  // Use Vercel's `ai` package to setup a stream
  const { stream, handlers } = LangChainStream();

  // Initialize Pinecone Client
  const pinecone = new Pinecone();

  const pineconeIndex = pinecone.Index(
    process.env.PINECONE_INDEX_NAME as string
  );

  // Initialize our vector store
  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_KEY }),
    { pineconeIndex: pineconeIndex, textKey: "text" }
  );

  console.log(vectorStore.embeddings);

  // Specify the OpenAI model we'd like to use, and turn on streaming
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo",
    streaming: true,
    callbacks: [handlers],
    openAIApiKey: process.env.OPENAI_KEY,
  });

  // Define the Langchain chain
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    {
      returnSourceDocuments: true,
      memory: new BufferMemory({
        memoryKey: "chat_history",
        inputKey: "question", // The key for the input to the chain
        outputKey: "text", // The key for the final conversational output of the chain
      }),
    }
  );

  // Get a streaming response from our chain with the prompt given by the user
  chain.stream({ question: body.prompt });

  // Return an output stream to the frontend
  return new StreamingTextResponse(stream);
}
