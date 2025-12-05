import Editor from '@monaco-editor/react';
import { Loader2 } from 'lucide-react';

interface QueryEditorProps {
  value: string;
  onChange: (value: string) => void;
  height?: string;
}

const defaultQuery = `query GetUser {
  user(id: 123) {
    id
    name
    email
    posts {
      id
      title
      comments {
        id
        content
      }
    }
  }
}`;

export function QueryEditor({ value, onChange, height = "400px" }: QueryEditorProps) {
  return (
    <div className="rounded-xl overflow-hidden border border-border/50 shadow-sm bg-code">
      <div className="flex items-center justify-between px-4 py-2 bg-code border-b border-border/20">
        <span className="text-xs font-medium text-code-foreground/70">GraphQL Query</span>
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-danger/60" />
          <div className="w-3 h-3 rounded-full bg-warning/60" />
          <div className="w-3 h-3 rounded-full bg-success/60" />
        </div>
      </div>
      <Editor
        height={height}
        defaultLanguage="graphql"
        value={value || defaultQuery}
        onChange={(v) => onChange(v || '')}
        theme="vs-dark"
        loading={
          <div className="flex items-center justify-center h-full bg-code">
            <Loader2 className="w-6 h-6 animate-spin text-primary" />
          </div>
        }
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          fontFamily: 'JetBrains Mono, monospace',
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          padding: { top: 16, bottom: 16 },
          automaticLayout: true,
          tabSize: 2,
          wordWrap: 'on',
          bracketPairColorization: { enabled: true },
        }}
      />
    </div>
  );
}
