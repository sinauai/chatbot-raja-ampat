import { type CoreMessage, generateText } from "ai"
import { google } from "@ai-sdk/google"
import newsData from "@/data/news.json"
import { getGeminiEmbedding, cosineSimilarity } from "@/lib/gemini-embedding"

// In-memory cache for news embeddings
let newsWithEmbeddings: (typeof newsData[0] & { embedding: number[] })[] | null = null;

async function ensureNewsEmbeddings() {
  if (newsWithEmbeddings) return newsWithEmbeddings;
  // Compute embedding for each article (full_text)
  newsWithEmbeddings = await Promise.all(
    newsData.map(async (article) => ({
      ...article,
      embedding: (await getGeminiEmbedding(article.full_text)) ?? []
    }))
  );
  return newsWithEmbeddings;
}

export async function POST(req: Request) {
  const { messages }: { messages: CoreMessage[] } = await req.json()
  const userQuestion = messages[messages.length - 1]?.content || ""

  // 1. Ensure news embeddings are ready
  const articles = await ensureNewsEmbeddings()

  // 2. Get embedding for user question (force string, handle possible array of parts)
  let userQuestionText = "";
  if (typeof userQuestion === "string") {
    userQuestionText = userQuestion;
  } else if (Array.isArray(userQuestion)) {
    userQuestionText = userQuestion.map(p => {
      if (typeof p === "string") return p;
      if (typeof p === "object" && p !== null && "text" in p && typeof (p as any).text === "string") return (p as any).text;
      return "";
    }).join(" ");
  } 
  const questionEmbedding = await getGeminiEmbedding(userQuestionText)

  // 3. Compute similarity and pick top 3 articles
  const topArticles = articles
    .map((article, idx) => ({
      ...article,
      similarity: questionEmbedding && article.embedding.length
        ? cosineSimilarity(questionEmbedding, article.embedding)
        : -1,
      originalIndex: idx
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 3)
    .sort((a, b) => a.originalIndex - b.originalIndex) // keep original order for ARTIKEL X

  // 4. Build newsContext from top articles only
  const newsContext = topArticles
    .map((article, idx) => `ARTIKEL ${article.originalIndex + 1}:
JUDUL: ${article.title}
URL: ${article.url}

${article.full_text}`)
    .join("\n\n---\n\n")

  const result = await generateText({
    model: google("gemini-2.0-flash"),
    system: `Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan berita di Kompas.id.
    
    Anda HANYA boleh menjawab pertanyaan berdasarkan informasi yang terdapat dalam konteks berita berikut:
    
    ${newsContext}
    
    Jika pertanyaan tidak terkait dengan informasi dalam konteks berita, jawablah "Maaf, saya tidak memiliki informasi tentang itu." 
    
    PENTING:
    1. Format jawaban Anda dalam Markdown yang rapi untuk meningkatkan keterbacaan.
    2. Gunakan paragraf, poin-poin, dan penekanan (bold/italic) dengan tepat.
    3. Elaborasi jawaban Anda dengan baik, tetapi tetap sesuai konteks pertanyaan.
    4. JANGAN menyertakan referensi seperti "(ARTIKEL X)" dalam jawaban Anda. Pengguna sudah dapat melihat sumber informasi di bagian terpisah.
    5. Tetap gunakan informasi dari artikel yang relevan, tetapi jangan menyebutkan nomor artikelnya dalam teks jawaban.
    6. Untuk keperluan internal sistem, tetap sertakan kode artikel yang Anda gunakan di AKHIR jawaban Anda dengan format: "ARTIKEL 1 ARTIKEL 2" (jika Anda menggunakan artikel 1 dan 2). Kode ini akan dihapus sebelum ditampilkan kepada pengguna.
    
    Jawablah dalam Bahasa Indonesia yang baik dan benar.`,
    messages,
    temperature: 0.7,
  })

  return new Response(result.text, {
    headers: { "Content-Type": "text/plain" },
  })
}