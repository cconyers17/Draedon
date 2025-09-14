import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Draedon - Text-to-CAD Architecture',
  description: 'Professional text-to-CAD architecture application with advanced NLP and 3D visualization',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}