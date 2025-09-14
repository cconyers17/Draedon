# Deployment Guide for Text-to-CAD Architecture Application

## Quick Start for Render Deployment

### Prerequisites
1. GitHub repository with your code
2. Render account (free tier available)
3. Code pushed to main branch

### 1. Deploy Backend (FastAPI)

1. **Create Web Service on Render:**
   - Go to https://render.com/dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select the repository and branch

2. **Configure Backend Service:**
   ```
   Name: text-to-cad-backend
   Runtime: Python 3
   Build Command: pip install --upgrade pip && pip install -r requirements.txt
   Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   Root Directory: backend
   Plan: Starter (Free)
   ```

3. **Environment Variables:**
   Add these in Render dashboard:
   ```
   DEBUG=False
   RENDER=True
   SECRET_KEY=[Generate in Render]
   CORS_ORIGINS=https://your-frontend-url.onrender.com
   SPACY_MODEL=en_core_web_sm
   NLP_CACHE_TTL=3600
   MAX_FILE_SIZE=104857600
   ```

4. **Create Database Services:**
   - Create PostgreSQL service (Starter plan - free)
   - Create Redis service (Starter plan - free)
   - Connect the connection strings to your web service

### 2. Deploy Frontend (Next.js)

1. **Create Web Service on Render:**
   - Click "New" → "Web Service"
   - Connect same GitHub repository
   - Select repository and branch

2. **Configure Frontend Service:**
   ```
   Name: text-to-cad-frontend
   Runtime: Node
   Build Command: npm ci && npm run build
   Start Command: npm start
   Root Directory: text-to-cad-app
   Plan: Starter (Free)
   ```

3. **Environment Variables:**
   ```
   NODE_ENV=production
   NEXT_PUBLIC_API_URL=https://text-to-cad-backend.onrender.com
   NEXT_PUBLIC_APP_ENV=production
   ```

### 3. Post-Deployment Configuration

1. **Update CORS Origins:**
   - Get your frontend URL from Render
   - Update backend CORS_ORIGINS environment variable
   - Redeploy backend

2. **Test the Application:**
   - Visit your frontend URL
   - Test basic functionality
   - Check browser console for errors
   - Verify API connectivity

### 4. Custom Domain (Optional)

1. **Frontend Custom Domain:**
   - In Render dashboard → Settings → Custom Domains
   - Add your domain
   - Update DNS records

2. **Backend Custom Domain:**
   - Add custom domain for API
   - Update frontend NEXT_PUBLIC_API_URL

## Production Optimizations

### Performance
- Enable Render's CDN for static assets
- Set up proper caching headers
- Monitor performance metrics

### Security
- Set up proper CORS origins
- Use HTTPS only
- Set secure environment variables
- Enable rate limiting

### Monitoring
- Set up Sentry for error tracking
- Monitor application logs
- Set up health checks

## Troubleshooting

### Common Issues

1. **Build Failures:**
   - Check Node.js/Python versions
   - Verify dependencies in package.json/requirements.txt
   - Check build logs in Render dashboard

2. **Runtime Errors:**
   - Check application logs
   - Verify environment variables
   - Test database connectivity

3. **CORS Issues:**
   - Verify CORS_ORIGINS includes your frontend URL
   - Check protocol (http vs https)
   - Ensure no trailing slashes

4. **WebAssembly Issues:**
   - Verify WASM files are properly served
   - Check Content-Type headers
   - Enable required security headers

### Useful Commands

```bash
# Test backend locally
cd backend
uvicorn app.main:app --reload

# Test frontend locally
cd text-to-cad-app
npm run dev

# Build frontend for production
npm run build
npm start
```

## Free Tier Limitations

### Render Free Tier:
- 750 hours/month compute time
- Services spin down after 15 minutes of inactivity
- Cold start delays (10-30 seconds)
- 100GB bandwidth/month
- 1GB RAM per service

### Recommendations:
- Keep services active with health checks
- Optimize cold start performance
- Monitor usage to avoid limits
- Consider upgrading for production use

## Support

For issues:
1. Check Render documentation
2. Review application logs
3. Test locally first
4. Check GitHub issues