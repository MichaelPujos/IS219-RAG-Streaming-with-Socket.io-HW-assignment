const express = require('express');
const http = require('http');
const { Server } = require("socket.io");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { Document } = require("langchain/document");
require("dotenv").config();

const app = express();
const server = http.createServer(app);
const io = new Server(server);
const PORT = process.env.PORT || 3003;

app.use(express.static(__dirname));

const docs = [
  new Document({ pageContent: "LangChain enables retrieval augmented generation by integrating tools and memory." }),
  new Document({ pageContent: "OpenAI embeddings convert text into vectors to allow semantic search." })
];

let vectorStore;
async function setupStore() {
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 100, chunkOverlap: 10 });
  const splitDocs = await splitter.splitDocuments(docs);
  const embeddings = new OpenAIEmbeddings();
  vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
}
setupStore();

io.on('connection', (socket) => {
  socket.on('query', async (msg) => {
    const results = await vectorStore.similaritySearch(msg, 1);
    const response = results.length ? results[0].pageContent : "No relevant data found.";
    io.emit('chat message', "RAG: " + response);
  });
});

server.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
