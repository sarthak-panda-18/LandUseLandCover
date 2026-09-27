# LULC Classification - Frontend Dashboard

React + Vite single-page application for interacting with the Land Use and Land Cover (LULC) classification system.

---

## Directory Structure

```
frontend/
├── public/                 # Static public assets (icons, images)
├── src/
│   ├── components/         # Reusable UI widgets and map components
│   ├── pages/              # Primary route pages / views
│   ├── services/           # HTTP API client services
│   ├── App.jsx             # Root layout and placeholder dashboard shell
│   ├── App.css             # Component-level styling
│   ├── index.css           # Global theme variables and typography
│   └── main.jsx            # React root DOM mounting point
├── index.html              # HTML template
├── package.json            # NPM scripts & dependencies
├── vite.config.js          # Vite bundler configuration
└── README.md               # This documentation
```

---

## Setup & Development

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Development Server
```bash
npm run dev
```
Open `http://localhost:5173` in your browser.

### 3. Build for Production
```bash
npm run build
```
