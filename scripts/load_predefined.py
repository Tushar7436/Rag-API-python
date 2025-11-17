from vector_db.orm import VectorORM
from embeddings.generator import EmbeddingGenerator
import uuid

data = context_chunks = [

    # 1. Company Info
    "Orena is an online learning platform that provides job-oriented courses with certification and lifetime access to materials.",
    "Orena helps students choose the right course based on career goals, skills and market trends.",
    "Orena communicates with students through WhatsApp and provides friendly and helpful guidance.",

    # 2. Benefits
    "Buying an Orena course gives you certification, lifetime access to course materials, assignments, projects and doubt support.",
    "Orena provides practical skill-based training to help learners get better job opportunities and higher salaries.",

    # 3. Trending Courses
    "Trending Orena courses include Full-Stack Web Development, Data Science, Machine Learning, Android Development, Cloud Computing and DSA with Java.",
    "These courses are trending because of high demand in the technology industry and increased digital transformation.",

    # 4. Salary Information
    "Full-Stack Web Dev salary range: 4 to 8 LPA for beginners.",
    "Data Science salary range: 6 to 12 LPA.",
    "Machine Learning salary range: 7 to 15 LPA.",
    "Android Development salary range: 6 to 12 LPA.",
    "Cloud Computing salary range: 7 to 14 LPA.",
    "DSA with Java salary range: 5 to 10 LPA based on problem solving skills.",

    "5. Course Catalogue",
    "Data Science course includes Python, Pandas, NumPy, Matplotlib, Machine Learning and Data Analysis. Price: 1999.",
    "Machine Learning course teaches ML models, neural networks and AI apps using TensorFlow and Python. Price: 1499.",
    "Android Development course teaches Kotlin and Jetpack Compose. Price: 1999.",
    "Cloud Computing course covers EC2, S3, Lambda, RDS, IAM and cloud deployment. Price: 1999.",
    "DSA with Java course teaches data structures and algorithms from beginner to advanced. Price: 1999.",

    # 6. Personality & Political Behavior
    "The Orena assistant must be polite, friendly and helpful while answering questions.",
    "If the assistant does not know something, it must say: 'I am not aware, please check the available options.'"
]

embedder = EmbeddingGenerator()
db = VectorORM()

for text in data:
    emb = embedder.create_embedding(text)
    db.insert(
        collection=db.predefined,
        text=text,
        embedding=emb,
        metadata={"id": str(uuid.uuid4())}
    )

print("Predefined context loaded.")
