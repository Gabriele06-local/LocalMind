use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::{Frame, Terminal};
use syntect::easy::HighlightLines;
use syntect::highlighting::{ThemeSet, Style as SynStyle};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

use crate::embed::Embedder;
use crate::index::Index;
use crate::search::top_k;

struct SearchState {
    results: Vec<crate::search::Scored>,
    status: String,
    loading: bool,
    searched_query: String,
}

struct PreviewCache {
    for_selected: usize,
    header: String,
    lines: Vec<Line<'static>>,
    ss: SyntaxSet,
    ts: ThemeSet,
}

impl PreviewCache {
    fn new() -> Self {
        Self {
            for_selected: usize::MAX,
            header: String::new(),
            lines: Vec::new(),
            ss: SyntaxSet::load_defaults_newlines(),
            ts: ThemeSet::load_defaults(),
        }
    }

    fn load(&mut self, selected: usize, path: &str, query: &str) {
        if self.for_selected == selected {
            return;
        }
        self.for_selected = selected;
        self.lines.clear();
        self.header.clear();

        let p = Path::new(path);
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("txt");

        let text = match std::fs::read_to_string(p) {
            Ok(t) => t,
            Err(_) => {
                self.lines.push(Line::from(Span::raw("[error reading file]")));
                return;
            }
        };

        self.header = format!(" {} ", p.file_name().unwrap_or_default().to_string_lossy());

        let syntax = self
            .ss
            .find_syntax_by_extension(ext)
            .unwrap_or_else(|| self.ss.find_syntax_plain_text());
        let theme = &self.ts.themes["base16-ocean.dark"];
        let mut h = HighlightLines::new(syntax, theme);

        let q_lower = query.to_lowercase();
        let q_terms: Vec<&str> = q_lower.split_whitespace().collect();

        let max_lines = 500;
        let mut count = 0;
        for line in LinesWithEndings::from(&text) {
            if count >= max_lines {
                break;
            }
            if let Ok(ranges) = h.highlight_line(line, &self.ss) {
                let spans: Vec<Span> = ranges
                    .iter()
                    .map(|(style, text)| {
                        let s = syn_style(style);
                        let trimmed = text.trim_end_matches('\n').trim_end_matches('\r');
                        if trimmed.is_empty() {
                            return Span::raw(" ");
                        }
                        let lower = trimmed.to_lowercase();
                        let has_match = query.len() > 1
                            && q_terms.iter().any(|t| lower.contains(t));
                        if has_match {
                            Span::styled(
                                trimmed.to_string(),
                                s.bg(Color::Rgb(60, 56, 6)).add_modifier(Modifier::BOLD),
                            )
                        } else {
                            Span::styled(trimmed.to_string(), s)
                        }
                    })
                    .collect();
                self.lines.push(Line::from(spans));
            }
            count += 1;
        }
        if text.len() > max_lines * 80 {
            self.lines.push(Line::from(Span::styled(
                format!("… {} more bytes", text.len() - max_lines * 80),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    fn clear(&mut self) {
        self.for_selected = usize::MAX;
        self.lines.clear();
        self.header.clear();
    }
}

fn syn_style(s: &SynStyle) -> Style {
    let fg = s.foreground;
    Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b))
}

pub fn run(
    embedder: Arc<Embedder>,
    index: Arc<RwLock<Index>>,
    progress_rx: std::sync::mpsc::Receiver<String>,
) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;

        let state = Arc::new(Mutex::new(SearchState {
            results: Vec::new(),
            status: "Type a query to search".into(),
            loading: false,
            searched_query: String::new(),
        }));

        let mut query = String::new();
        let mut cursor: usize = 0;
        let mut selected: usize = 0;
        let mut last_key = Instant::now();
        let debounce = std::time::Duration::from_millis(200);
        let mut preview = PreviewCache::new();

        let res = loop {
            {
                let s = state.lock().unwrap();
                if let Some(r) = s.results.get(selected) {
                    preview.load(selected, &r.path, &query);
                } else {
                    preview.clear();
                }
            }

            terminal.draw(|f| draw_ui(f, &query, cursor, &state, selected, &preview))?;

            while let Ok(msg) = progress_rx.try_recv() {
                let mut s = state.lock().unwrap();
                if !s.loading && s.searched_query.is_empty() && query.is_empty() {
                    s.status = msg;
                }
            }

            if event::poll(std::time::Duration::from_millis(30))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                        KeyCode::Esc
                        | KeyCode::Char('c')
                            if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) =>
                        {
                            break Ok(());
                        }
                        KeyCode::Esc => {
                            if query.is_empty() {
                                break Ok(());
                            }
                            query.clear();
                            cursor = 0;
                            last_key = Instant::now();
                        }
                        KeyCode::Enter => {
                            let s = state.lock().unwrap();
                            if let Some(r) = s.results.get(selected) {
                                let _ = std::process::Command::new("cmd")
                                    .args(["/C", "start", "", &r.path])
                                    .spawn();
                            }
                        }
                        KeyCode::Char(c) => {
                            query.insert(cursor, c);
                            cursor += 1;
                            last_key = Instant::now();
                        }
                        KeyCode::Backspace => {
                            if cursor > 0 {
                                cursor -= 1;
                                query.remove(cursor);
                                last_key = Instant::now();
                            }
                        }
                        KeyCode::Delete => {
                            if cursor < query.len() {
                                query.remove(cursor);
                                last_key = Instant::now();
                            }
                        }
                        KeyCode::Left => {
                            cursor = cursor.saturating_sub(1);
                        }
                        KeyCode::Right => {
                            cursor = cursor.min(query.len().saturating_add(1));
                            if cursor > query.len() {
                                cursor = query.len();
                            }
                        }
                        KeyCode::Home => cursor = 0,
                        KeyCode::End => cursor = query.len(),
                        KeyCode::Up => {
                            selected = selected.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            let s = state.lock().unwrap();
                            if selected + 1 < s.results.len() {
                                selected += 1;
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            let elapsed = last_key.elapsed();
            if !query.is_empty() && elapsed >= debounce {
                let mut s = state.lock().unwrap();
                if s.searched_query != query && !s.loading {
                    s.searched_query = query.clone();
                    s.loading = true;
                    s.status = "Searching...".into();
                    drop(s);

                    let shared = state.clone();
                    let q = query.clone();
                    let emb = embedder.clone();
                    let idx = index.clone();
                    tokio::task::spawn_blocking(move || {
                        let start = Instant::now();
                        if let Ok(embedding) = emb.embed(&q) {
                            let guard = idx.read().unwrap();
                            let results = top_k(&guard, &embedding, 10);
                            drop(guard);
                            let elapsed = start.elapsed();
                            let mut s = shared.lock().unwrap();
                            s.results = results;
                            s.loading = false;
                            s.status = format!(
                                "{} results ({:.2}s)",
                                s.results.len(),
                                elapsed.as_secs_f32()
                            );
                        }
                    });
                }
            }
        };

        let _ = disable_raw_mode();
        let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
        let _ = terminal.show_cursor();
        res
    })
}

fn draw_ui(
    f: &mut Frame,
    query: &str,
    cursor: usize,
    state: &Arc<Mutex<SearchState>>,
    selected: usize,
    preview: &PreviewCache,
) {
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(f.area());

    let s = state.lock().unwrap();

    let search_input = Paragraph::new(Line::from(Span::raw(format!("> {}", query))))
        .block(Block::default().borders(Borders::ALL).title(" Search "))
        .style(if s.loading {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::White)
        });
    f.render_widget(search_input, areas[0]);
    f.set_cursor_position((areas[0].x + 2 + cursor as u16, areas[0].y + 1));

    let content_area = areas[1];
    let horiz = Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(content_area);

    // ── Results list (left) ──
    let items: Vec<ListItem> = s
        .results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let score_color = if r.score > 0.5 {
                Color::Green
            } else if r.score > 0.3 {
                Color::Yellow
            } else {
                Color::DarkGray
            };
            let content = Line::from(vec![
                Span::styled(
                    format!(" {:>3}. ", i + 1),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{:.3} ", r.score),
                    Style::default().fg(score_color),
                ),
                Span::raw(&r.path),
            ]);
            if i == selected {
                ListItem::new(content).style(
                    Style::default()
                        .bg(Color::Blue)
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ListItem::new(content)
            }
        })
        .collect();

    let results_list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Results "))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));
    f.render_widget(results_list, horiz[0]);

    // ── Preview panel (right) ──
    let preview_block = Block::default().borders(Borders::ALL).title(preview.header.as_str());

    if preview.lines.is_empty() {
        let empty = Paragraph::new(Line::from(Span::styled(
            " Select a result to preview ",
            Style::default().fg(Color::DarkGray),
        )))
        .block(preview_block);
        f.render_widget(empty, horiz[1]);
    } else {
        let visible: Vec<Line> = {
            let max_h = preview_block.inner(horiz[1]).height as usize;
            preview.lines.iter().take(max_h).cloned().collect()
        };
        let p = Paragraph::new(visible).block(preview_block);
        f.render_widget(p, horiz[1]);
    }

    let status_bar = Paragraph::new(Line::from(Span::raw(&s.status)))
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(status_bar, areas[2]);
}
