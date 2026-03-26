# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (from root)
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Development: server runs at http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) chatbot for course materials. The system uses Anthropic Claude with tool-based semantic search against ChromaDB.

### Data Flow

```
docs/*.txt → DocumentProcessor → Course/Lesson/Chunk objects
                              ↓
                         VectorStore (ChromaDB)
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
            course_catalog       course_content
            (metadata, titles)   (chunked embeddings)
```

### Core Components

**RAGSystem** (`rag_system.py`) - Main orchestrator that coordinates:
- Document processing and ingestion
- Query processing with Claude tools
- Session management

**Query Processing Pipeline:**
1. User query → API (`/api/query`)
2. RAGSystem checks conversation history via SessionManager
3. AIGenerator calls Claude with `CourseSearchTool` available
4. If Claude needs course content: `ToolManager.execute_tool("search_course_content")`
5. VectorStore performs semantic search with optional filters (course_name, lesson_number)
6. Results formatted and returned to Claude for final answer
7. Sources extracted from tool, response saved to session

**Document Format** (`docs/*.txt`):
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]
Lesson N: [lesson title]
[lesson content...]
```

### ChromaDB Collections

- **course_catalog**: Stores course metadata (title, instructor, lessons JSON). Used for semantic course name resolution when user queries reference partial course names.
- **course_content**: Stores chunked course material (800 chars/chunk, 100 overlap). Metadata includes `course_title`, `lesson_number`, `chunk_index`.

### Tool System

`CourseSearchTool` implements the `Tool` abstract base class. It provides:
- `search_course_content(query, course_name?, lesson_number?)` function to Claude
- Returns formatted results with course/lesson headers
- Tracks `last_sources` for UI display

### Configuration (`backend/config.py`)

Key settings:
- `CHUNK_SIZE: 800`, `CHUNK_OVERLAP: 100` - document chunking
- `MAX_RESULTS: 5` - search results per query
- `MAX_HISTORY: 2` - conversation history depth
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"` - semantic embeddings
