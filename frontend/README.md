# Text-to-CAD Frontend

Next.js frontend for the Text-to-CAD Architecture Application.

## Note for Deployment

The complete frontend source code is located in the `text-to-cad-app` directory. This `frontend` directory contains the essential configuration files for deployment reference.

## Complete Frontend Structure

```
text-to-cad-app/
├── src/
│   ├── app/                  # Next.js App Router pages
│   │   ├── complexity/       # Complexity level pages (L0-L3)
│   │   └── page.tsx         # Home page
│   ├── components/          # React components
│   │   ├── chat/           # Chat interface for text input
│   │   ├── viewer/         # 3D CAD viewers
│   │   ├── materials/      # Material management
│   │   └── ui/             # UI components
│   ├── lib/                # Core libraries
│   │   ├── cad/           # CAD engine and OpenCASCADE
│   │   ├── nlp/           # NLP processing
│   │   └── materials/     # Material database
│   └── types/             # TypeScript definitions
├── public/                # Static assets
└── Configuration files    # package.json, next.config.ts, etc.
```

## Deployment Instructions

For Render deployment, use the complete `text-to-cad-app` directory as your build source.

### Build Commands:
```bash
npm ci
npm run build
```

### Start Command:
```bash
npm start
```

### Environment Variables:
```
NODE_ENV=production
NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
NEXT_PUBLIC_APP_ENV=production
```

## Features

- **3D CAD Visualization**: Three.js + OpenCASCADE.js
- **Complexity Levels**: L0 (Basic) to L3 (Iconic Structures)
- **Real-time Chat Interface**: Natural language to CAD conversion
- **Material Management**: Comprehensive building materials database
- **Export Capabilities**: Multiple CAD formats (STEP, STL, OBJ, etc.)
- **Professional UI**: Modern, responsive design with Tailwind CSS

## Technology Stack

- **Next.js 15** with App Router
- **React 18** + TypeScript
- **Three.js** for 3D rendering
- **OpenCASCADE.js** for CAD operations
- **Tailwind CSS** for styling
- **WebAssembly** for high-performance CAD

## Development

```bash
cd text-to-cad-app
npm install
npm run dev
```

Visit `http://localhost:3000` to see the application.