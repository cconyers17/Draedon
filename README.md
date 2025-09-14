# Text-to-CAD Architecture Web Application

A sophisticated web application that converts natural language architectural descriptions into professional 3D CAD models with real-time visualization.

## 🏗️ **Features**

- **Natural Language Processing**: Convert architectural descriptions to CAD models
- **4 Complexity Levels**: L0 (Basic) to L3 (Iconic Structures)
- **3D Visualization**: Real-time Three.js rendering with OpenCASCADE.js
- **Multiple Export Formats**: STEP, STL, OBJ, IFC, DXF
- **Professional Architecture**: Next.js frontend + FastAPI backend
- **Material Database**: Comprehensive building materials and properties
- **Building Code Validation**: Automatic compliance checking

## 🚀 **Live Demo**

- **Frontend**: [Deploy to Render](https://render.com)
- **Backend API**: [Deploy to Render](https://render.com)

## 🛠️ **Technology Stack**

### Frontend
- **Next.js 15** with App Router
- **React 18** + TypeScript
- **Three.js** for 3D visualization
- **OpenCASCADE.js** for CAD operations
- **Tailwind CSS** for styling

### Backend
- **FastAPI** with Python 3.11+
- **OpenCASCADE** for CAD geometry
- **spaCy + Transformers** for NLP
- **PostgreSQL** + **Redis** for data
- **Trimesh** for mesh processing

## 📁 **Project Structure**

```
├── text-to-cad-app/          # Next.js Frontend
│   ├── src/
│   │   ├── app/              # App Router pages
│   │   ├── components/       # React components
│   │   ├── lib/              # Utilities & CAD engine
│   │   └── types/            # TypeScript definitions
│   ├── public/               # Static assets
│   └── package.json
├── backend/                  # FastAPI Backend
│   ├── app/
│   │   ├── api/              # API routes
│   │   ├── services/         # Business logic
│   │   │   ├── cad/          # CAD generation
│   │   │   └── nlp/          # NLP processing
│   │   ├── models/           # Database models
│   │   └── core/             # Configuration
│   └── requirements.txt
├── documentation/            # Technical docs
└── DEPLOYMENT_GUIDE.md      # Deployment instructions
```

## 🔧 **Local Development**

### Prerequisites
- Node.js 18+
- Python 3.11+
- Git

### Frontend Setup
```bash
cd text-to-cad-app
npm install
npm run dev
```

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 🚀 **Deployment**

### Render Platform (Recommended)

1. **Fork this repository**
2. **Create Render services**:
   - Web Service for frontend (Node.js)
   - Web Service for backend (Python)
   - PostgreSQL database
   - Redis cache

3. **Configure environment variables** (see `DEPLOYMENT_GUIDE.md`)
4. **Deploy** - Services will auto-deploy from GitHub

### Manual Deployment
See `DEPLOYMENT_GUIDE.md` for detailed instructions.

## 📊 **Complexity Levels**

| Level | Description | Features | Use Cases |
|-------|-------------|----------|-----------|
| **L0** | Basic Shapes | Simple geometry, single materials | Learning, prototyping |
| **L1** | Residential | Multi-room houses, building codes | Home design, small projects |
| **L2** | Commercial | Multi-story, MEP systems | Office buildings, retail |
| **L3** | Iconic | Parametric, custom algorithms | Landmarks, architectural art |

## 🔬 **Example Usage**

```
Input: "Create a 3-bedroom house with kitchen and 2 bathrooms, 150 square meters"

Processing:
- NLP extracts: 3 bedrooms, 1 kitchen, 2 bathrooms, 150m²
- CAD generates: Floor plan + 3D model
- Validation: Building code compliance
- Export: STEP, STL, IFC formats

Output: Professional CAD model ready for construction
```

## 🛡️ **Architecture Highlights**

- **Microservices**: Separate frontend/backend for scalability
- **Type Safety**: Full TypeScript + Python type hints
- **Caching**: Redis for NLP results and CAD operations
- **Error Handling**: Comprehensive logging and monitoring
- **Security**: Rate limiting, CORS, input validation
- **Performance**: WebAssembly CAD engine, optimized 3D rendering

## 📖 **Documentation**

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Complete setup instructions
- [API Documentation](backend/README.md) - Backend API reference
- [Frontend Guide](text-to-cad-app/README.md) - Component documentation
- [Architecture Overview](documentation/) - Technical deep-dive

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- OpenCASCADE for CAD geometry engine
- Three.js community for 3D visualization
- spaCy team for NLP capabilities
- Next.js team for the amazing framework

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/yourusername/text-to-cad/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/text-to-cad/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/text-to-cad/wiki)

---

**Built with ❤️ for architects, engineers, and design professionals**