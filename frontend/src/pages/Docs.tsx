import { Book, Code, Shield, Zap, ExternalLink } from 'lucide-react';

const docSections = [
  {
    icon: Shield,
    title: 'Security Analysis',
    description: 'Learn how GQLGuard detects malicious patterns, introspection attacks, and query abuse.',
    link: '#',
  },
  {
    icon: Code,
    title: 'API Reference',
    description: 'Integrate GQLGuard into your GraphQL server with our REST API endpoints.',
    link: '#',
  },
  {
    icon: Zap,
    title: 'Quick Start',
    description: 'Get up and running in minutes with our step-by-step integration guide.',
    link: '#',
  },
];

const Docs = () => {
  return (
    <div className="min-h-screen bg-background pt-20 pb-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-5">
            <Book className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-3">
            Documentation
          </h1>
          <p className="text-muted-foreground max-w-lg mx-auto">
            Everything you need to integrate and configure GQLGuard for your GraphQL applications.
          </p>
        </div>
        
        {/* Doc Sections */}
        <div className="grid sm:grid-cols-3 gap-4 mb-12">
          {docSections.map((section, idx) => {
            const Icon = section.icon;
            return (
              <a
                key={idx}
                href={section.link}
                className="glass-card-elevated rounded-xl p-6 hover:border-primary/30 transition-all group"
              >
                <div className="w-11 h-11 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <Icon className="w-5 h-5 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                  {section.title}
                  <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                </h3>
                <p className="text-sm text-muted-foreground">
                  {section.description}
                </p>
              </a>
            );
          })}
        </div>
        
        {/* Coming Soon Notice */}
        <div className="glass-card rounded-xl p-8 text-center">
          <p className="text-muted-foreground">
            Full documentation is coming soon. In the meantime, explore the analyzer and dashboard.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Docs;
