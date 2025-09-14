# GitHub Setup Instructions

## Step 1: Create GitHub Repository

1. **Go to GitHub**: Visit https://github.com
2. **Create New Repository**:
   - Click the "+" icon → "New repository"
   - Repository name: `Draedon`
   - Description: "Professional text-to-CAD architecture application with Next.js frontend and FastAPI backend"
   - Set to **Public** (required for Render free tier)
   - **DO NOT** initialize with README (we already have one)
   - Click "Create repository"

## Step 2: Connect Local Repository to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/Draedon.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify Upload

1. Go to your GitHub repository page
2. Verify you see all these key directories:
   - `backend/` - FastAPI backend
   - `frontend/` - Next.js config files
   - `text-to-cad-app/` - Complete frontend source
   - `DEPLOYMENT_GUIDE.md` - Deployment instructions
   - `README.md` - Project overview

## Step 4: Deploy to Render

### Backend Deployment:
1. **Create Web Service** on Render
2. **Connect GitHub repo**
3. **Settings**:
   ```
   Name: text-to-cad-backend
   Root Directory: backend
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
4. **Add Environment Variables**:
   ```
   DEBUG=False
   RENDER=True
   SECRET_KEY=[Auto-generated]
   CORS_ORIGINS=https://your-frontend-url.onrender.com
   ```

### Frontend Deployment:
1. **Create Web Service** on Render
2. **Connect same GitHub repo**
3. **Settings**:
   ```
   Name: text-to-cad-frontend
   Root Directory: text-to-cad-app
   Runtime: Node
   Build Command: npm ci && npm run build
   Start Command: npm start
   ```
4. **Add Environment Variables**:
   ```
   NODE_ENV=production
   NEXT_PUBLIC_API_URL=https://text-to-cad-backend.onrender.com
   ```

### Database Services:
1. **Create PostgreSQL** service (free tier)
2. **Create Redis** service (free tier)
3. **Connect URLs** to backend environment

## Step 5: Update CORS

After deployment:
1. Get your frontend URL from Render
2. Update backend `CORS_ORIGINS` environment variable
3. Redeploy backend

## Important Notes

- **Repository must be PUBLIC** for Render free tier
- **Use the complete file paths** shown above for Root Directory
- **Frontend source is in `text-to-cad-app/`** directory
- **Backend source is in `backend/`** directory
- See `DEPLOYMENT_GUIDE.md` for detailed instructions

## Troubleshooting

- **Build fails**: Check Node.js/Python versions in build logs
- **CORS errors**: Verify CORS_ORIGINS includes your frontend URL
- **WebAssembly issues**: Ensure proper headers are configured
- **Database connection**: Verify PostgreSQL service is connected

Your sophisticated text-to-CAD application will be live once deployed! 🚀