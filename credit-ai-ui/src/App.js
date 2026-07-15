import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  CreditCard, Zap, TrendingUp, Clock, Wallet, LayoutDashboard, User, X,
  CheckCircle, Plus, Trash2, ShieldCheck, ChevronRight, Store, Gift,
  Database, Filter, Lock, Unlock, Users, Send, Bot, Search, GitCompare,
  Brain, Sparkles, AlertCircle, RefreshCw, Info, ArrowRight, BarChart2
} from 'lucide-react';

const API = "http://127.0.0.1:8000";

const CATEGORY_ORDER = ["dining", "travel", "shopping", "fuel", "utility", "others"];
const CAT_EMOJIS = { dining:"🍽️", travel:"✈️", shopping:"🛍️", fuel:"⛽", utility:"💡", others:"📦" };
const CAT_COLORS = {
  dining:   { bg:"bg-orange-50",  border:"border-orange-200",  text:"text-orange-600",  badge:"bg-orange-100",  bar:"bg-orange-400" },
  travel:   { bg:"bg-sky-50",     border:"border-sky-200",     text:"text-sky-600",     badge:"bg-sky-100",     bar:"bg-sky-400"    },
  shopping: { bg:"bg-violet-50",  border:"border-violet-200",  text:"text-violet-600",  badge:"bg-violet-100",  bar:"bg-violet-400" },
  fuel:     { bg:"bg-amber-50",   border:"border-amber-200",   text:"text-amber-600",   badge:"bg-amber-100",   bar:"bg-amber-400"  },
  utility:  { bg:"bg-teal-50",    border:"border-teal-200",    text:"text-teal-600",    badge:"bg-teal-100",    bar:"bg-teal-400"   },
  others:   { bg:"bg-slate-50",   border:"border-slate-200",   text:"text-slate-500",   badge:"bg-slate-100",   bar:"bg-slate-400"  },
};
const TIER_GRAD = {
  Basic:        "from-slate-500 to-slate-700",
  Mid:          "from-blue-500 to-blue-800",
  Premium:      "from-violet-600 to-purple-900",
  Super_Premium:"from-amber-500 to-yellow-700",
};
const NETWORK_COLOR = { Visa:"#1A1F71", Mastercard:"#EB001B", RuPay:"#1E88E5", Diners:"#004A97", Amex:"#007BC1" };

const SWIPE_MERCHANTS = [
  'Amazon','Flipkart','Zomato','Swiggy','Uber','Ola','IndiGo',
  'MakeMyTrip','Indian Oil','Shell','Airtel','Netflix','Starbucks',
  'KFC','Myntra','Apollo Pharmacy','LIC Insurance','BPCL','HPCL',
  'Spotify','Rapido','Zerodha','BookMyShow','Dominos','Ajio','Decathlon'
];

const BankLogo = ({ bankName, size="sm" }) => {
  const sz = size === "lg" ? "w-10 h-10" : size === "xl" ? "w-14 h-14" : "w-6 h-6";
  return (
    <img src={`/logos/${bankName}.png`} alt={bankName}
      className={`${sz} rounded-full object-contain bg-white border border-white/30 p-0.5 flex-shrink-0`}
      onError={e => { e.target.onerror=null; e.target.src=`https://ui-avatars.com/api/?name=${bankName}&background=random&color=fff&size=128`; }}
    />
  );
};

// ─────────────────────────────────────────────
// VIRTUAL CARD VISUAL
// ─────────────────────────────────────────────
const CardVisual = ({ card }) => {
  const tier = card.card_tier || "Basic";
  const grad = TIER_GRAD[tier] || TIER_GRAD.Basic;
  const network = card.network || "Visa";
  const netColor = NETWORK_COLOR[network] || "#1A1F71";

  return (
    <div className={`relative w-full aspect-[1.586/1] rounded-2xl bg-gradient-to-br ${grad} p-5 text-white overflow-hidden shadow-2xl select-none`}>
      {/* Background circles */}
      <div className="absolute -right-8 -top-8 w-44 h-44 bg-white/10 rounded-full" />
      <div className="absolute -right-4 bottom-0 w-28 h-28 bg-white/5 rounded-full" />
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-60 h-60 bg-white/5 rounded-full" />

      {/* Chip */}
      <div className="absolute top-5 left-5 w-9 h-7 bg-amber-300/80 rounded-md flex items-center justify-center z-10">
        <div className="w-6 h-5 border border-amber-500/50 rounded-sm grid grid-cols-2 gap-px p-px">
          {[...Array(4)].map((_, i) => <div key={i} className="bg-amber-500/30 rounded-sm" />)}
        </div>
      </div>

      {/* Bank logo top right */}
      <div className="absolute top-4 right-4 z-10">
        <BankLogo bankName={card.bank_name} size="lg" />
      </div>

      {/* Card number (masked) */}
      <div className="absolute bottom-16 left-5 z-10">
        <p className="text-white/40 text-[9px] uppercase font-bold tracking-widest mb-1">Card Number</p>
        <p className="font-mono text-sm tracking-[0.25em] text-white/80">•••• •••• •••• 4242</p>
      </div>

      {/* Card name bottom */}
      <div className="absolute bottom-5 left-5 right-16 z-10">
        <p className="text-white/50 text-[8px] uppercase tracking-widest mb-0.5">Card Name</p>
        <p className="font-black text-sm leading-tight truncate">{card.card_name}</p>
        <p className="text-white/60 text-[9px] font-bold uppercase tracking-wider">{card.bank_name} • {tier.replace('_', ' ')}</p>
      </div>

      {/* Network logo bottom right */}
      <div className="absolute bottom-4 right-4 z-10">
        <div className="text-right">
          <p className="text-[10px] font-black uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.7)" }}>{network}</p>
          {network === "Mastercard" && (
            <div className="flex -space-x-2 mt-0.5 justify-end">
              <div className="w-5 h-5 rounded-full bg-red-500 opacity-90" />
              <div className="w-5 h-5 rounded-full bg-amber-400 opacity-90" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// CARD DETAIL MODAL
// ─────────────────────────────────────────────
const CardModal = ({ card, onClose }) => {
  if (!card) return null;
  const lounge = parseInt(card.lounge_access) === 1;
  const loungeType = card.lounge_type && card.lounge_type !== "None" && card.lounge_type !== "0" ? card.lounge_type : null;
  const rateEntries = Object.entries(card.rates || {}).sort((a, b) => b[1] - a[1]);
  const topCat = rateEntries[0]?.[0];

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-white rounded-3xl max-w-xl w-full shadow-2xl overflow-hidden max-h-[92vh] flex flex-col" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className={`bg-gradient-to-br ${TIER_GRAD[card.card_tier] || TIER_GRAD.Basic} p-6 flex-shrink-0`}>
          <button onClick={onClose} className="absolute top-4 right-4 bg-white/20 hover:bg-white/30 p-1.5 rounded-full z-10">
            <X size={15} className="text-white" />
          </button>
          {/* Virtual card */}
          <div className="max-w-[280px] mx-auto">
            <CardVisual card={card} />
          </div>
          {/* Savings highlight */}
          <div className="mt-4 bg-white/15 rounded-2xl p-4 flex justify-between items-center">
            <div>
              <p className="text-white/60 text-[9px] uppercase font-bold">Est. Annual Savings</p>
              <p className="text-2xl font-black text-white">₹{(card.annual_savings||0).toLocaleString()}</p>
            </div>
            <div className="text-right">
              <p className="text-white/60 text-[9px] uppercase font-bold">AI Match</p>
              <p className="text-2xl font-black text-white">{card.match}%</p>
            </div>
            <div className="text-right">
              <p className="text-white/60 text-[9px] uppercase font-bold">Tier</p>
              <p className="text-sm font-black text-white">{card.card_tier?.replace('_',' ')}</p>
            </div>
          </div>
        </div>

        {/* Body */}
        <div className="overflow-y-auto flex-1 p-6 space-y-5">
          {/* Fees */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label:"Annual Fee",   value: card.annual_fee==0 ? "Free" : `₹${Number(card.annual_fee).toLocaleString()}` },
              { label:"Fee Waiver",   value: card.fee_waiver_spend>0 ? `₹${Number(card.fee_waiver_spend).toLocaleString()}` : "N/A" },
              { label:"Min. Income",  value: card.min_income>0 ? `₹${Number(card.min_income).toLocaleString()}` : "None" },
            ].map(({ label, value }) => (
              <div key={label} className="bg-slate-50 border border-slate-100 rounded-2xl p-3 text-center">
                <p className="text-[9px] uppercase font-bold text-slate-400 mb-1">{label}</p>
                <p className="font-black text-slate-800 text-sm">{value}</p>
              </div>
            ))}
          </div>

          {/* Reward rates */}
          <div>
            <h3 className="text-xs font-black text-slate-600 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Sparkles size={13} className="text-violet-500" /> Reward Rates
            </h3>
            <div className="grid grid-cols-3 gap-2">
              {rateEntries.map(([cat, rate]) => {
                const col = CAT_COLORS[cat] || CAT_COLORS.others;
                const isTop = cat === topCat;
                const cap = Number(card.caps?.[cat] || 0);
                return (
                  <div key={cat} className={`${col.bg} ${col.border} border rounded-2xl p-3 text-center relative ${isTop ? 'ring-2 ring-offset-1 ring-violet-400' : ''}`}>
                    {isTop && <span className="absolute -top-2 left-1/2 -translate-x-1/2 bg-violet-500 text-white text-[7px] font-black uppercase px-1.5 py-0.5 rounded-full whitespace-nowrap">Best</span>}
                    <p className="text-lg mb-0.5">{CAT_EMOJIS[cat]}</p>
                    <p className={`text-[8px] uppercase font-bold ${col.text} mb-0.5`}>{cat}</p>
                    <p className={`text-base font-black ${col.text}`}>{rate}%</p>
                    <p className="text-[8px] font-semibold text-slate-400 mt-0.5">
                      {Number(rate) === 0 ? 'no rewards' : cap > 0 ? `cap ₹${cap.toLocaleString()}/mo` : 'uncapped'}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Perks */}
          <div>
            <h3 className="text-xs font-black text-slate-600 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Gift size={13} className="text-emerald-500" /> Perks & Benefits
            </h3>
            <div className="space-y-2">
              {(card.highlights ? card.highlights.split('|') : []).map((h, i) => (
                <div key={i} className="flex items-start gap-3 bg-slate-50 rounded-xl p-3">
                  <CheckCircle size={14} className="text-emerald-500 mt-0.5 flex-shrink-0" />
                  <p className="text-xs font-semibold text-slate-700">{h}</p>
                </div>
              ))}
              {card.welcome_benefit && card.welcome_benefit !== "None" && (
                <div className="flex items-start gap-3 bg-violet-50 rounded-xl p-3">
                  <Gift size={14} className="text-violet-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-[9px] uppercase font-bold text-violet-400">Welcome Benefit</p>
                    <p className="text-xs font-semibold text-slate-700">{card.welcome_benefit}</p>
                  </div>
                </div>
              )}
              {card.milestone_benefit && card.milestone_benefit !== "None" && (
                <div className="flex items-start gap-3 bg-amber-50 rounded-xl p-3">
                  <Sparkles size={14} className="text-amber-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-[9px] uppercase font-bold text-amber-500">Milestone Benefits</p>
                    <p className="text-xs font-semibold text-slate-700">{card.milestone_benefit}</p>
                  </div>
                </div>
              )}
              <div className={`flex items-start gap-3 rounded-xl p-3 ${lounge ? 'bg-sky-50' : 'bg-slate-50'}`}>
                {lounge
                  ? <CheckCircle size={14} className="text-sky-500 mt-0.5 flex-shrink-0" />
                  : <X size={14} className="text-slate-300 mt-0.5 flex-shrink-0" />}
                <div>
                  <p className="text-sm font-bold text-slate-700">Airport Lounge Access</p>
                  {lounge
                    ? <p className="text-xs text-sky-600 font-semibold">
                        {Number(card.lounge_domestic_yr) >= 99
                          ? 'Unlimited domestic & international visits'
                          : `${card.lounge_domestic_yr || 0} domestic${Number(card.lounge_intl_yr) > 0 ? ` + ${card.lounge_intl_yr} international` : ''} visits/year`}
                        {loungeType ? ` • ${loungeType}` : ''}
                      </p>
                    : <p className="text-xs text-slate-400">Not available on this card</p>}
                </div>
              </div>
              <div className="flex items-start gap-3 bg-slate-50 rounded-xl p-3">
                <CheckCircle size={14} className="text-violet-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-bold text-slate-700">Eligibility: {card.eligibility_level || "Standard"}</p>
                  <p className="text-xs text-slate-400">
                    {card.network} network • {card.card_category?.replace('_', ' ')}
                    {card.reward_type ? ` • rewards as ${card.reward_type}` : ''}
                    {card.point_value_rs && card.reward_type !== 'cashback' ? ` (1 pt ≈ ₹${card.point_value_rs})` : ''}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Data source */}
          {card.source_url && (
            <p className="text-[9px] text-slate-300 text-center">
              Real card data, verified {card.data_updated || 'recently'} •{' '}
              <a href={card.source_url} target="_blank" rel="noreferrer" className="underline hover:text-slate-400">source</a>
            </p>
          )}

          {/* Income eligibility banner */}
          {card.income_eligible === false && (
            <div className="bg-red-50 border border-red-200 rounded-2xl p-4 flex gap-3">
              <AlertCircle size={14} className="text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-black text-red-700 mb-1">⚠ Income Eligibility Not Met</p>
                <p className="text-xs text-red-600">
                  This card requires a minimum annual income of ₹{Number(card.min_income).toLocaleString()}.
                  Your current income does not qualify yet — but this card is shown so you know what to aim for.
                </p>
              </div>
            </div>
          )}

          {/* Tip */}
          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 flex gap-3">
            <AlertCircle size={14} className="text-amber-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-amber-700 font-medium">
              {card.annual_fee==0
                ? "Lifetime free — no spend target needed. Pure upside."
                : card.fee_waiver_spend>0
                  ? `Spend ₹${Number(card.fee_waiver_spend).toLocaleString()} to waive the ₹${Number(card.annual_fee).toLocaleString()} fee. At your savings estimate, this card pays for itself.`
                  : `Fixed fee of ₹${Number(card.annual_fee).toLocaleString()} — make sure your rewards exceed this.`}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// COMPARE MODAL — unlimited cards, table layout
// ─────────────────────────────────────────────
const CompareModal = ({ cards, onClose }) => {
  if (!cards || cards.length < 2) return null;
  const categories = CATEGORY_ORDER;

  const getBestIdx = (values, higher=true) => {
    if (values.every(v => v === values[0])) return -1;
    return higher ? values.indexOf(Math.max(...values)) : values.indexOf(Math.min(...values));
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-white rounded-3xl max-w-5xl w-full shadow-2xl overflow-hidden max-h-[92vh] flex flex-col" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="bg-slate-900 p-6 flex-shrink-0">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-slate-400 text-[10px] uppercase font-bold mb-1">Side-by-Side Analysis</p>
              <h2 className="text-white font-black text-xl">Comparing {cards.length} Cards</h2>
            </div>
            <button onClick={onClose} className="bg-white/10 hover:bg-white/20 p-2 rounded-xl transition-colors"><X size={16} className="text-white" /></button>
          </div>
          {/* Card thumbnails */}
          <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${cards.length}, 1fr)` }}>
            {cards.map(card => (
              <div key={card.card_name} className="bg-white/10 rounded-2xl p-3 text-center">
                <BankLogo bankName={card.bank_name} size="lg" />
                <p className="text-white font-black text-xs mt-2 leading-tight">{card.card_name}</p>
                <p className="text-2xl font-black text-emerald-400 mt-1">₹{(card.annual_savings||0).toLocaleString()}</p>
                <p className="text-slate-400 text-[8px] uppercase">Est. Savings</p>
              </div>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="overflow-auto flex-1 p-6">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr>
                <th className="text-left text-slate-400 text-[10px] uppercase font-bold pb-3 pr-4 w-32">Attribute</th>
                {cards.map(c => (
                  <th key={c.card_name} className="text-center text-slate-700 text-xs font-black pb-3 px-2">{c.card_name}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {/* Annual Fee */}
              {[
                { label:"Annual Fee",        vals: cards.map(c => Number(c.annual_fee||0)),         higher:false, fmt:v => v===0?"Free":`₹${v.toLocaleString()}` },
                { label:"Fee Waiver",         vals: cards.map(c => Number(c.fee_waiver_spend||0)),   higher:false, fmt:v => v>0?`₹${v.toLocaleString()}`:"N/A" },
                { label:"Min. Income",        vals: cards.map(c => Number(c.min_income||0)),         higher:false, fmt:v => v>0?`₹${v.toLocaleString()}`:"Any" },
                { label:"AI Match Score",     vals: cards.map(c => c.match||0),                     higher:true,  fmt:v => `${v}%` },
                { label:"Est. Annual Savings",vals: cards.map(c => c.annual_savings||0),            higher:true,  fmt:v => `₹${v.toLocaleString()}` },
              ].map(({ label, vals, higher, fmt }) => {
                const bestIdx = getBestIdx(vals, higher);
                return (
                  <tr key={label} className="hover:bg-slate-50">
                    <td className="text-slate-500 font-bold text-[10px] uppercase py-2.5 pr-4">{label}</td>
                    {vals.map((v, i) => (
                      <td key={i} className={`text-center font-black text-sm py-2.5 px-2 ${i===bestIdx?'text-emerald-600':'text-slate-400'}`}>
                        {fmt(v)}
                        {i===bestIdx && <span className="ml-1 text-[8px] bg-emerald-100 text-emerald-600 px-1 py-0.5 rounded">best</span>}
                      </td>
                    ))}
                  </tr>
                );
              })}

              {/* Category rates */}
              <tr><td colSpan={cards.length+1} className="pt-4 pb-2"><p className="text-[10px] uppercase font-black text-slate-400">Reward Rates by Category</p></td></tr>
              {categories.map(cat => {
                const col = CAT_COLORS[cat] || CAT_COLORS.others;
                const vals = cards.map(c => Number(c.rates?.[cat]||0));
                const bestIdx = getBestIdx(vals, true);
                return (
                  <tr key={cat} className="hover:bg-slate-50">
                    <td className={`text-[10px] uppercase font-bold py-2 pr-4 ${col.text}`}>{CAT_EMOJIS[cat]} {cat}</td>
                    {vals.map((v, i) => (
                      <td key={i} className={`text-center font-black text-sm py-2 px-2 ${i===bestIdx?'text-emerald-600':v===0?'text-slate-200':'text-slate-400'}`}>
                        {v}%
                        {i===bestIdx && v>0 && <span className="ml-1 text-[8px] bg-emerald-100 text-emerald-600 px-1 py-0.5 rounded">best</span>}
                      </td>
                    ))}
                  </tr>
                );
              })}

              {/* Lounge */}
              <tr><td colSpan={cards.length+1} className="pt-4 pb-2"><p className="text-[10px] uppercase font-black text-slate-400">Perks</p></td></tr>
              {[
                { label:"Lounge Access", vals: cards.map(c => parseInt(c.lounge_access)===1 ? (c.lounge_type&&c.lounge_type!=="None"?c.lounge_type:"Yes") : "None") },
                { label:"Best Category", vals: cards.map(c => c.max_benefit||"—") },
                { label:"Card Tier",     vals: cards.map(c => c.card_tier?.replace('_',' ')||"—") },
              ].map(({ label, vals }) => (
                <tr key={label} className="hover:bg-slate-50">
                  <td className="text-slate-500 font-bold text-[10px] uppercase py-2.5 pr-4">{label}</td>
                  {vals.map((v, i) => (
                    <td key={i} className={`text-center text-xs font-black py-2.5 px-2 ${v!=="None"&&v!=="—"?'text-emerald-600':'text-slate-300'}`}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-center text-slate-400 text-xs font-bold mt-4">Green = better value in that row</p>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────
const App = () => {
  const [users, setUsers] = useState([]);
  const [currentUser, setCurrentUser] = useState('');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [allCardsList, setAllCardsList] = useState([]);
  const [selectedBankFilter, setSelectedBankFilter] = useState('All');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState("");
  const [sliders, setSliders] = useState({ dining:0,travel:0,shopping:0,fuel:0,utility:0,others:0 });
  const [manualRecs, setManualRecs] = useState([]);
  const [swipeResult, setSwipeResult] = useState(null);
  const [swipeLoading, setSwipeLoading] = useState(false);
  const [swipeSearch, setSwipeSearch] = useState("");
  const [selectedCard, setSelectedCard] = useState(null);
  const [newCardSelect, setNewCardSelect] = useState("");
  const [compareCards, setCompareCards] = useState([]);
  const [showCompare, setShowCompare] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState([{ role:'ai', content:"Hey! I'm the NeuroCard AI. Ask me anything about your rewards or which card to use." }]);
  const [isChatting, setIsChatting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const chatEndRef = useRef(null);
  const [newUserParams, setNewUserParams] = useState({ name:'',password:'',persona:'foodie',income:1000000,redemption_pref:'Cashback',owned_cards:[] });
  const [isGenerating, setIsGenerating] = useState(false);
  const [rewardsAnalysis, setRewardsAnalysis] = useState(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const fileInputRef = useRef(null);

  const fetchUsers = async () => {
    try {
      const res = await axios.get(`${API}/users`);
      setUsers(res.data);
      if (res.data.length===0) { setCurrentUser('+ Add New User'); setIsAuthenticated(true); setLoading(false); }
      else if (!currentUser) { setCurrentUser(res.data[0]); setIsAuthenticated(false); setLoading(false); }
    } catch(e) { console.error(e); setCurrentUser('+ Add New User'); setIsAuthenticated(true); setLoading(false); }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post(`${API}/login`, { user_id:currentUser, password:loginPassword });
      if (res.data.success) { setIsAuthenticated(true); setLoginError(""); setLoginPassword(""); fetchProfile(currentUser); }
      else setLoginError(res.data.error||"Invalid Password");
    } catch(err) { console.error(err); setLoginError("Server error. Is the backend running?"); }
  };

  const handleUserSwitch = (u) => {
    setCurrentUser(u); setData(null);
    if (u==='+ Add New User'||u==='+ Manage Users') setIsAuthenticated(true);
    else { setIsAuthenticated(false); setLoginError(""); setLoginPassword(""); }
  };

  const handleDeleteUser = async (u) => {
    if (!window.confirm(`Delete ${u}?`)) return;
    try { await axios.delete(`${API}/delete-user/${u}`); fetchUsers(); } catch(e) { console.error(e); }
  };

  const fetchProfile = async (name) => {
    if (name==='+ Add New User') { setData(null); setCurrentUser(name); return; }
    setLoading(true);
    try {
      const res = await axios.get(`${API}/profile/${name}`);
      if (res.data && !res.data.error) {
        setData(res.data);
        setSliders(res.data.summary);
        const recs = await axios.post(`${API}/manual-analyze`, { user_id:name, ...res.data.summary });
        setManualRecs(recs.data.recommendations);
        setSwipeResult(null);
        setChatHistory([{ role:'ai', content:`Hey ${name}! I'm your NeuroCard AI. What would you like to know about your rewards?` }]);
      }
    } catch(err) { console.error(err); }
    finally { setLoading(false); }
  };

  const initializeApp = async () => {
    try {
      const res = await axios.get(`${API}/all-cards-list`);
      setAllCardsList(res.data);
      if (res.data.length>0) setNewCardSelect(res.data[0].card_name);
    } catch(e) { console.error(e); }
    fetchUsers();
  };

  useEffect(() => { initializeApp(); }, []);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior:"smooth" }); }, [chatHistory]);

  const handleCreateUser = async (e) => {
    e.preventDefault();
    if (!newUserParams.name.trim()) return alert("Enter a name.");
    if (!newUserParams.password.trim()) return alert("Create a password.");
    if (newUserParams.owned_cards.length===0) return alert("Select at least one card.");
    setIsGenerating(true);
    try {
      await axios.post(`${API}/create-user`, newUserParams);
      await fetchUsers();
      setCurrentUser(newUserParams.name); setIsAuthenticated(true);
      fetchProfile(newUserParams.name);
      setNewUserParams({ name:'',password:'',persona:'foodie',income:1000000,redemption_pref:'Cashback',owned_cards:[] });
      setActiveTab('dashboard');
    } catch(err) { console.error(err); alert("Failed to create user. Is the backend running?"); }
    finally { setIsGenerating(false); }
  };

  const toggleOnboardingCard = (cn) => {
    const c = newUserParams.owned_cards;
    setNewUserParams({ ...newUserParams, owned_cards: c.includes(cn) ? c.filter(x=>x!==cn) : [...c, cn] });
  };

  const fetchRewardsAnalysis = async (name) => {
    try {
      const r = await axios.get(`${API}/rewards-analysis/${name}`);
      if (!r.data.error) setRewardsAnalysis(r.data.cards);
    } catch(e) { console.error(e); }
  };
  useEffect(() => {
    if (activeTab==='rewards' && currentUser && data) fetchRewardsAnalysis(currentUser);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, currentUser, data]);

  const handleStatementUpload = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setUploadBusy(true); setUploadResult(null);
    try {
      const fd = new FormData();
      fd.append('user_id', currentUser);
      fd.append('file', f);
      fd.append('replace', 'true');
      const r = await axios.post(`${API}/upload-statement`, fd);
      if (r.data.error) setUploadResult({ error: r.data.error });
      else { setUploadResult(r.data); fetchProfile(currentUser); }
    } catch(err) { console.error(err); setUploadResult({ error: 'Upload failed — is the backend running?' }); }
    finally { setUploadBusy(false); e.target.value=''; }
  };

  const handleCategoryCorrection = async (merchant, category) => {
    try {
      await axios.post(`${API}/set-category`, { user_id: currentUser, merchant, category });
      fetchProfile(currentUser);   // reclassify everything with the new rule
    } catch(e) { console.error(e); }
  };

  const handleSliderChange = (cat, val) => setSliders({ ...sliders, [cat]:parseInt(val) });
  const runManualAI = async () => {
    try {
      const res = await axios.post(`${API}/manual-analyze`, { user_id:currentUser, ...sliders });
      setManualRecs(res.data.recommendations);
    } catch(err) { console.error(err); }
  };

  const runSwipe = async (m) => {
    const merchant = m||swipeSearch.trim();
    if (!merchant) { alert("Enter a brand name"); return; }
    setSwipeLoading(true); setSwipeResult(null);
    try {
      const res = await axios.get(`${API}/smart-swipe?user=${currentUser}&merchant=${encodeURIComponent(merchant)}`);
      setSwipeResult({ ...res.data, merchant });
    } catch(err) { console.error(err); setSwipeResult({ error:"Could not reach server." }); }
    finally { setSwipeLoading(false); }
  };

  const addToWallet = async () => {
    if (!newCardSelect) return;
    try {
      const res = await axios.post(`${API}/wallet/add`, { user_id:currentUser, card_name:newCardSelect });
      setData({ ...data, owned_cards:res.data.owned_cards });
    } catch(err) { console.error(err); }
  };

  const removeFromWallet = async (cn) => {
    try {
      const res = await axios.post(`${API}/wallet/remove`, { user_id:currentUser, card_name:cn });
      setData({ ...data, owned_cards:res.data.owned_cards });
      setSwipeResult(null);
    } catch(err) { console.error(err); }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    const newHistory = [...chatHistory, { role:'user', content:chatInput }];
    setChatHistory(newHistory); setChatInput(""); setIsChatting(true);
    try {
      const res = await axios.post(`${API}/api/advisor/chat`, { user_id:currentUser, message:chatInput });
      setChatHistory([...newHistory, { role:'ai', content:res.data.reply }]);
    } catch(err) {
      console.error(err);
      setChatHistory([...newHistory, { role:'ai', content:"Couldn't reach the AI. Try again!" }]);
    }
    setIsChatting(false);
  };

  // Unlimited compare — toggle card in/out of compare list
  const toggleCompare = (card) => {
    setCompareCards(prev =>
      prev.find(c=>c.card_name===card.card_name) ? prev.filter(c=>c.card_name!==card.card_name) : [...prev, card]
    );
  };

  const handleRefreshCards = async () => {
    setIsRefreshing(true);
    try {
      const res = await axios.post(`${API}/refresh-cards`);
      alert(res.data.message);
      initializeApp();
    } catch(err) { console.error(err); alert("Refresh failed."); }
    finally { setIsRefreshing(false); }
  };

  const uniqueBanks = ['All', ...new Set(allCardsList.map(c=>c.bank_name))];

  const SOURCE_BADGE = {
    rules:  { label:"Keyword Match", color:"bg-emerald-500" },
    ml:     { label:"AI Classified",  color:"bg-blue-500" },
    gemini: { label:"Gemini AI",      color:"bg-violet-500" },
  };

  if (loading && !['+ Add New User','+ Manage Users'].includes(currentUser) && isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center gap-4">
        <Brain className="text-blue-500 animate-pulse" size={48} />
        <p className="text-slate-400 font-bold text-sm tracking-widest uppercase">NeuroCard is thinking...</p>
      </div>
    );
  }

  const renderCardGrid = (recList, isManual=false) => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {recList.slice(0,4).map((card, idx) => {
        const inCompare = !!compareCards.find(c=>c.card_name===card.card_name);
        return (
          <div key={idx} className="bg-white border border-slate-200 p-5 rounded-[24px] hover:border-blue-300 hover:shadow-lg transition-all group relative">
            {isManual && (
              <button
                onClick={e => { e.stopPropagation(); toggleCompare(card); }}
                className={`absolute top-4 right-4 text-[9px] font-black uppercase px-2 py-1 rounded-lg border flex items-center gap-1 transition-all
                  ${inCompare ? 'bg-violet-600 text-white border-violet-600' : 'bg-white text-slate-400 border-slate-200 hover:border-violet-400 hover:text-violet-500'}`}
              >
                <GitCompare size={10} /> {inCompare ? 'Added ✓' : 'Compare'}
              </button>
            )}
            <div onClick={() => setSelectedCard(card)} className="cursor-pointer">
              <div className="flex items-center gap-3 mb-4 pr-20">
                <BankLogo bankName={card.bank_name} />
                <div>
                  <h4 className="font-black text-slate-800 text-sm leading-tight">{card.card_name}</h4>
                  <p className="text-[9px] text-slate-400 font-bold uppercase">{card.bank_name} • {card.network}</p>
                </div>
              </div>
              <div className="flex items-end justify-between">
                <div>
                  <p className="text-[9px] font-bold uppercase text-slate-400">Est. Savings</p>
                  <p className="text-2xl font-black text-slate-800">₹{(card.annual_savings||0).toLocaleString()}</p>
                </div>
                <div className="text-right">
                  <span className="text-[10px] font-black uppercase text-emerald-600 bg-emerald-50 px-2 py-1 rounded-lg">
                    {isManual?'Simulated':'AI Match'}: {card.match}%
                  </span>
                  {parseInt(card.lounge_access)===1 && (
                    <p className="text-[9px] text-sky-500 font-bold mt-1">
                      ✈ {card.lounge_type&&card.lounge_type!=="None"?card.lounge_type:"Lounge"}
                    </p>
                  )}
                </div>
              </div>
              {card.income_eligible === false && (
                <p className="text-[9px] font-bold text-red-500 mt-2 flex items-center gap-1">⚠ Income req. not met</p>
              )}
              <p className="text-[10px] text-slate-400 font-bold uppercase mt-1 flex items-center gap-1">View details <ChevronRight size={10} /></p>
            </div>
          </div>
        );
      })}
    </div>
  );

  return (
    <div className="min-h-screen bg-[#f0f4f9] text-slate-900 font-sans p-4 md:p-8">
      {selectedCard && <CardModal card={selectedCard} onClose={() => setSelectedCard(null)} />}
      {showCompare && compareCards.length >= 2 && <CompareModal cards={compareCards} onClose={() => setShowCompare(false)} />}

      {/* HEADER */}
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center mb-8 gap-4">
        <h1 className="text-3xl font-black tracking-tight text-slate-800 flex items-center gap-3">
          <Brain className="text-blue-600" size={32} />
          Neuro<span className="text-blue-600">Card</span>
          <span className="text-xs font-bold text-slate-400 bg-slate-100 px-2 py-1 rounded-lg ml-1 tracking-widest uppercase">AI</span>
        </h1>
        <div className="flex items-center gap-3">
          <button onClick={handleRefreshCards} disabled={isRefreshing}
            className="bg-white border border-slate-200 text-slate-600 hover:text-blue-600 hover:border-blue-300 px-3 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all disabled:opacity-50"
            title="Refresh card data from web">
            <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
            {isRefreshing ? 'Refreshing...' : 'Refresh Cards'}
          </button>
          <div className="bg-white px-4 py-2 rounded-xl shadow-sm border border-slate-200 flex items-center gap-3">
            <User size={18} className="text-blue-500" />
            <select value={currentUser} onChange={e=>handleUserSwitch(e.target.value)} className="bg-transparent font-bold text-slate-700 outline-none cursor-pointer">
              <optgroup label="Select Profile">{users.map(u=><option key={u} value={u}>{u}</option>)}</optgroup>
              <optgroup label="Options">
                <option value="+ Add New User">+ Add New User</option>
                <option value="+ Manage Users">+ Manage Users</option>
              </optgroup>
            </select>
          </div>
        </div>
      </div>

      {/* VIEWS */}
      {currentUser==='+ Manage Users' ? (
        <div className="max-w-3xl mx-auto bg-white border border-slate-200 p-10 rounded-[40px] shadow-xl">
          <div className="flex items-center gap-4 mb-8 border-b pb-4">
            <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center text-red-600"><Users size={24}/></div>
            <h2 className="text-2xl font-black text-slate-800">Manage Users</h2>
          </div>
          {users.length===0 ? <p className="text-slate-500 text-center py-10 font-bold">No users yet.</p> :
            <div className="space-y-4">{users.map(u=>(
              <div key={u} className="flex justify-between items-center bg-slate-50 p-5 rounded-2xl border border-slate-100">
                <p className="font-black text-slate-800 text-lg">{u}</p>
                <button onClick={()=>handleDeleteUser(u)} className="bg-red-100 hover:bg-red-600 text-red-600 hover:text-white px-4 py-2 rounded-xl font-bold flex items-center gap-2 transition-colors">
                  <Trash2 size={16}/> Delete
                </button>
              </div>
            ))}</div>}
        </div>

      ) : currentUser==='+ Add New User' ? (
        <div className="max-w-5xl mx-auto bg-white border border-slate-200 p-10 rounded-[40px] shadow-xl">
          <div className="text-center mb-10">
            <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4 text-blue-600"><Brain size={32}/></div>
            <h2 className="text-3xl font-black text-slate-800 mb-2">Create NeuroCard Profile</h2>
            <p className="text-slate-500">Set up your portfolio to activate the AI engine.</p>
          </div>
          <form onSubmit={handleCreateUser} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-xs font-bold uppercase text-slate-400 mb-2">Full Name</label>
                <input required type="text" value={newUserParams.name} onChange={e=>setNewUserParams({...newUserParams,name:e.target.value})} className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 font-bold text-slate-800 outline-none focus:border-blue-500" placeholder="e.g. Kalki" />
              </div>
              <div>
                <label className="block text-xs font-bold uppercase text-slate-400 mb-2">Password</label>
                <input required type="password" value={newUserParams.password} onChange={e=>setNewUserParams({...newUserParams,password:e.target.value})} className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 font-bold text-slate-800 outline-none focus:border-blue-500" placeholder="Create a password" />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-xs font-bold uppercase text-slate-400 mb-2">Annual Income (₹)</label>
                <input required type="number" value={newUserParams.income} onChange={e=>setNewUserParams({...newUserParams,income:parseFloat(e.target.value)})} className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 font-bold text-slate-800 outline-none focus:border-blue-500" />
              </div>
              <div>
                <label className="block text-xs font-bold uppercase text-slate-400 mb-2">Spending Persona</label>
                <select value={newUserParams.persona} onChange={e=>setNewUserParams({...newUserParams,persona:e.target.value})} className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 font-bold text-slate-800 outline-none focus:border-blue-500">
                  <option value="foodie">Foodie (Dining Heavy)</option>
                  <option value="traveler">Traveler (Flights & Hotels)</option>
                  <option value="shopper">Shopper (E-commerce)</option>
                  <option value="commuter">Commuter (Fuel & Transit)</option>
                  <option value="saver">Saver (Utilities & Bills)</option>
                  <option value="balanced">Balanced (Even Split)</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-xs font-bold uppercase text-slate-400 mb-2">Redemption Preference</label>
              <select value={newUserParams.redemption_pref} onChange={e=>setNewUserParams({...newUserParams,redemption_pref:e.target.value})} className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 font-bold text-slate-800 outline-none focus:border-blue-500">
                <option value="Cashback">Statement Cashback</option>
                <option value="Travel">Flights & Hotel Bookings</option>
                <option value="Shopping Vouchers">Shopping Vouchers (Amazon / Flipkart)</option>
                <option value="Dining Vouchers">Dining Vouchers (Zomato / Swiggy)</option>
                <option value="Fuel Surcharge">Fuel Surcharge Waiver</option>
                <option value="Investments">Gold / Mutual Fund Units</option>
                <option value="Movie & Events">Movie & Event Tickets</option>
                <option value="Lounge Access">Airport Lounge Visits</option>
              </select>
            </div>
            <div className="border-t border-slate-100 pt-6">
              <label className="block text-xs font-bold uppercase text-slate-400 mb-2">Cards You Own</label>
              <div className="flex items-center gap-3 mb-4 bg-slate-50 border border-slate-200 p-2 rounded-xl w-full md:w-1/2">
                <Filter size={16} className="text-slate-400 ml-2"/>
                <select value={selectedBankFilter} onChange={e=>setSelectedBankFilter(e.target.value)} className="bg-transparent w-full text-sm font-bold text-slate-700 outline-none">
                  {uniqueBanks.map(b=><option key={b} value={b}>{b==='All'?'All Banks':b}</option>)}
                </select>
              </div>
              <div className="max-h-64 overflow-y-auto border border-slate-200 rounded-2xl p-4 bg-slate-50 flex flex-wrap gap-2">
                {allCardsList.filter(c=>selectedBankFilter==='All'||c.bank_name===selectedBankFilter).map(c=>(
                  <button type="button" key={c.card_name} onClick={()=>toggleOnboardingCard(c.card_name)}
                    className={`px-3 py-2 text-[11px] font-bold rounded-xl border flex items-center gap-2 transition-all ${newUserParams.owned_cards.includes(c.card_name)?'bg-blue-600 text-white border-blue-600':'bg-white text-slate-600 border-slate-200 hover:border-blue-400'}`}>
                    <BankLogo bankName={c.bank_name}/> {c.card_name}
                  </button>
                ))}
              </div>
            </div>
            <div className="pt-4">
              <button type="submit" disabled={isGenerating||newUserParams.owned_cards.length===0} className="w-full bg-slate-900 hover:bg-slate-800 text-white font-black py-4 rounded-2xl flex items-center justify-center gap-3 disabled:opacity-50">
                {isGenerating ? <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"/> : <Database size={20}/>}
                {isGenerating ? 'Generating AI Profile...' : 'Create Profile & Import 12-Month History'}
              </button>
            </div>
          </form>
        </div>

      ) : !isAuthenticated ? (
        <div className="max-w-md mx-auto mt-20 bg-white border border-slate-200 p-10 rounded-[40px] shadow-2xl">
          <div className="w-20 h-20 bg-blue-50 rounded-3xl flex items-center justify-center mx-auto mb-6"><Lock className="text-blue-600" size={40}/></div>
          <h2 className="text-3xl font-black text-center text-slate-800 mb-2">Welcome Back</h2>
          <p className="text-center text-slate-500 font-medium mb-8">Enter password to access <strong>{currentUser}</strong>'s portfolio.</p>
          <form onSubmit={handleLogin}>
            <div className="mb-6">
              <input type="password" value={loginPassword} onChange={e=>setLoginPassword(e.target.value)} required autoFocus
                className="w-full bg-slate-50 border border-slate-200 rounded-2xl px-6 py-4 text-center tracking-widest text-2xl font-black text-slate-800 outline-none focus:border-blue-500" placeholder="••••"/>
              {loginError && <p className="text-red-500 text-xs font-bold text-center mt-3">{loginError}</p>}
            </div>
            <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white font-black py-4 rounded-2xl flex items-center justify-center gap-2">
              <Unlock size={18}/> Access Dashboard
            </button>
          </form>
        </div>

      ) : (
        data && (
          <>
            {/* TABS */}
            <div className="max-w-7xl mx-auto mb-8 border-b border-slate-200 flex gap-6 overflow-x-auto">
              {[
                { id:'dashboard', icon:LayoutDashboard, label:'Analytics' },
                { id:'rewards',   icon:Gift,            label:'Card ROI & Advisor' },
                { id:'wallet',    icon:Wallet,          label:'Wallet & Smart Swipe' },
                { id:'strategy',  icon:TrendingUp,      label:'AI Strategy Lab' },
              ].map(tab => (
                <button key={tab.id} onClick={()=>setActiveTab(tab.id)}
                  className={`flex items-center gap-2 pb-4 px-2 font-bold whitespace-nowrap border-b-2 transition-all ${activeTab===tab.id?'border-blue-600 text-blue-600':'border-transparent text-slate-500 hover:text-slate-800'}`}>
                  <tab.icon size={18}/> {tab.label}
                </button>
              ))}
            </div>

            {/* ── DASHBOARD ── */}
            {activeTab==='dashboard' && (
              <div className="max-w-7xl mx-auto">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-8">
                  <div className="lg:col-span-4 space-y-6">
                    <div className="bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500 p-8 rounded-[32px] shadow-xl text-white relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-4 opacity-10"><ShieldCheck size={120}/></div>
                      <div className="bg-white/20 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/30 flex items-center gap-2 w-fit mb-6 relative z-10">
                        <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"/>
                        <span className="text-[9px] font-black uppercase tracking-widest">NeuroCard AI Active</span>
                      </div>
                      <p className="text-white/70 text-[10px] font-bold uppercase tracking-[0.2em] mb-1 relative z-10">Your Spending Persona</p>
                      <h4 className="text-3xl font-black mb-1 tracking-tight leading-none relative z-10">{data.info.persona}</h4>
                      {data.info.persona_reason && (
                        <p className="text-white/60 text-[10px] font-bold mb-3 relative z-10">{data.info.persona_reason}</p>
                      )}
                      <div className="bg-black/10 backdrop-blur-sm p-4 rounded-2xl border border-white/10 relative z-10">
                        <p className="text-white/80 text-[10px] font-bold uppercase mb-2">💬 AI Insight</p>
                        <p className="text-white text-sm font-medium leading-relaxed">{data.info.description}</p>
                      </div>
                    </div>
                    <div className="bg-white border border-slate-200 p-8 rounded-[32px] shadow-sm">
                      <h3 className="text-slate-800 font-bold mb-6 flex items-center gap-2 text-lg">Spend Matrix</h3>
                      {(() => {
                        const maxVal = Math.max(...Object.values(data.summary), 1);
                        return CATEGORY_ORDER.map(cat => {
                          const col = CAT_COLORS[cat]||CAT_COLORS.others;
                          return (
                            <div key={cat} className="mb-4">
                              <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase mb-1.5">
                                <span>{CAT_EMOJIS[cat]} {cat}</span>
                                <span className="text-slate-700">₹{data.summary[cat]?.toLocaleString()||0}</span>
                              </div>
                              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div className={`h-full ${col.bar} rounded-full`} style={{ width:`${((data.summary[cat]||0)/maxVal)*100}%` }}/>
                              </div>
                            </div>
                          );
                        });
                      })()}
                    </div>
                  </div>
                  <div className="lg:col-span-8 space-y-6">
                    <div className="bg-blue-600 p-8 rounded-[32px] shadow-xl text-white flex flex-col md:flex-row justify-between items-center gap-6">
                      <div>
                        <p className="text-blue-200 text-[10px] uppercase font-bold tracking-widest mb-1">Income-Filtered Picks</p>
                        <h3 className="text-2xl font-black">Top Cards for {data.info.persona}</h3>
                        <p className="text-sm text-blue-100 mt-2">Based on ₹{(data.info.income/100000).toFixed(1)}L income and actual spend patterns.</p>
                      </div>
                      <div className="bg-white text-slate-900 p-6 rounded-3xl text-center min-w-[180px] border-4 border-blue-500">
                        <p className="text-[10px] text-slate-400 font-bold uppercase mb-1">Left on Table</p>
                        <p className="text-3xl font-black text-red-500">₹{data.retroactive_stats.missed_value.toLocaleString()}</p>
                        <p className="text-[9px] text-slate-400 font-bold mt-1">vs best possible</p>
                      </div>
                    </div>
                    {renderCardGrid(data.recommendations, false)}
                  </div>
                </div>
                <div className="bg-white border border-slate-200 p-8 rounded-[40px] shadow-sm mb-8">
                  <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
                    <h3 className="text-slate-800 font-bold flex items-center gap-2 text-xl"><Clock className="text-slate-400"/> Recent Transactions</h3>
                    <div className="flex items-center gap-3">
                      {uploadResult?.imported && <span className="text-xs font-bold text-emerald-600">✓ {uploadResult.imported} real transactions imported</span>}
                      {uploadResult?.error && <span className="text-xs font-bold text-red-500">{uploadResult.error}</span>}
                      <input ref={fileInputRef} type="file" accept=".csv,.xls,.xlsx" className="hidden" onChange={handleStatementUpload}/>
                      <button onClick={()=>fileInputRef.current?.click()} disabled={uploadBusy}
                        className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-xs font-black px-4 py-2.5 rounded-xl transition-all">
                        {uploadBusy ? 'Analyzing your statement…' : '📄 Upload Real Statement'}
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-[320px] overflow-y-auto pr-2">
                    {data.transactions?.map((t, i) => {
                      const col = CAT_COLORS[t.category]||CAT_COLORS.others;
                      return (
                        <div key={i} className="flex justify-between items-start p-4 bg-slate-50 rounded-2xl border border-transparent hover:border-slate-200 transition-all">
                          <div className="flex items-center gap-3">
                            <div className={`w-10 h-10 ${col.bg} ${col.border} border rounded-xl flex items-center justify-center text-lg flex-shrink-0`}>{CAT_EMOJIS[t.category]||'📦'}</div>
                            <div>
                              <p className="text-sm font-bold text-slate-800">{t.merchant}</p>
                              <p className="text-[9px] text-slate-400 font-black uppercase">{t.date}</p>
                              <select value={t.category} title="Wrong category? Fix it — the app remembers forever"
                                onChange={e=>handleCategoryCorrection(t.merchant, e.target.value)}
                                className={`text-[8px] uppercase font-bold ${col.text} ${col.badge} px-1 py-0.5 rounded-md mt-0.5 inline-block border-0 outline-none cursor-pointer appearance-auto`}>
                                {['dining','travel','shopping','fuel','utility','others'].map(c=><option key={c} value={c}>{c}</option>)}
                              </select>
                            </div>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <p className="text-sm font-black text-slate-800">₹{t.amount?.toLocaleString()}</p>
                            <p className="text-[9px] uppercase font-bold text-slate-400">{t.paid_using}</p>
                            <p className="text-[10px] font-black text-emerald-500">+{t.points_earned} pts</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* ── REWARDS & CHAT ── */}
            {activeTab==='rewards' && (
              <div className="max-w-7xl mx-auto">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  <div className="lg:col-span-5 space-y-6">
                    {/* Card ROI — is each card paying for itself? */}
                    <div className="bg-white border border-slate-200 rounded-[32px] p-6 shadow-sm">
                      <h3 className="font-black text-slate-800 text-lg mb-1 flex items-center gap-2">
                        <BarChart2 size={18} className="text-emerald-500"/> Card ROI Audit
                      </h3>
                      <p className="text-xs text-slate-400 font-medium mb-5">
                        Assumes smart routing: each spending category goes to whichever of YOUR cards earns the most on it.
                        Does every card then justify its fee?
                      </p>
                      {!rewardsAnalysis && <p className="text-xs text-slate-400">Analyzing your wallet…</p>}
                      {rewardsAnalysis && rewardsAnalysis.length === 0 && (
                        <p className="text-xs text-slate-400">Add cards to your wallet to see their ROI.</p>
                      )}
                      <div className="space-y-4">
                        {rewardsAnalysis?.map(c => (
                          <div key={c.card_name} className={`rounded-2xl border p-4 ${c.roi >= 0 ? 'bg-emerald-50/50 border-emerald-100' : 'bg-red-50/50 border-red-100'}`}>
                            <div className="flex justify-between items-start mb-2">
                              <div>
                                <p className="text-sm font-black text-slate-800">{c.card_name}</p>
                                <p className="text-[9px] uppercase font-bold text-slate-400">
                                  {c.bank_name} • {c.best_for?.length
                                    ? `best in wallet for: ${c.best_for.join(', ')}`
                                    : 'not the best pick for any category'}
                                </p>
                              </div>
                              <div className="text-right">
                                <p className={`text-lg font-black ${c.roi >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                                  {c.roi >= 0 ? '+' : '−'}₹{Math.abs(c.roi).toLocaleString()}
                                </p>
                                <p className="text-[8px] uppercase font-bold text-slate-400">per year, net of fee</p>
                              </div>
                            </div>
                            <div className="grid grid-cols-3 gap-2 mb-3">
                              {[
                                { label:'Would earn / yr', val:`₹${c.optimal_value.toLocaleString()}` },
                                { label:'Annual Fee', val: c.annual_fee ? `₹${c.annual_fee.toLocaleString()}` : 'Free' },
                                { label:'Spend routed here', val:`₹${c.routed_spend.toLocaleString()}` },
                              ].map(({label,val}) => (
                                <div key={label} className="bg-white rounded-xl p-2 text-center border border-slate-100">
                                  <p className="text-[8px] uppercase font-bold text-slate-400">{label}</p>
                                  <p className="text-xs font-black text-slate-700">{val}</p>
                                </div>
                              ))}
                            </div>
                            {c.fee_waiver_spend > 0 && (
                              <div className="mb-2">
                                <div className="flex justify-between text-[9px] font-bold text-slate-400 mb-1">
                                  <span>Fee-waiver progress</span>
                                  <span>{c.waiver_progress_pct}% of ₹{c.fee_waiver_spend.toLocaleString()}</span>
                                </div>
                                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                  <div className={`h-full rounded-full ${c.waiver_progress_pct >= 100 ? 'bg-emerald-400' : 'bg-amber-400'}`} style={{ width:`${c.waiver_progress_pct}%` }}/>
                                </div>
                              </div>
                            )}
                            <p className="text-[10px] font-bold text-slate-600">{c.verdict}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Projected annual rewards under smart routing */}
                    <div className="bg-white border border-slate-200 p-6 rounded-[32px] shadow-sm">
                      <h3 className="text-slate-800 font-bold mb-1 flex items-center gap-2">
                        <CreditCard size={16} className="text-blue-500"/> Projected Rewards by Card
                      </h3>
                      <p className="text-[10px] text-slate-400 font-medium mb-4">What each card earns per year if you follow the routing plan above.</p>
                      <div className="space-y-3">
                        {rewardsAnalysis?.filter(c=>c.optimal_value>0).map(c => {
                          const total = rewardsAnalysis.reduce((a,x)=>a+x.optimal_value,0)||1;
                          const pct = (c.optimal_value/total)*100;
                          return (
                            <div key={c.card_name} className="space-y-1">
                              <div className="flex justify-between text-xs">
                                <span className="font-bold text-slate-700 truncate max-w-[60%]">{c.card_name}</span>
                                <span className="font-black text-emerald-600">
                                  ₹{c.optimal_value.toLocaleString()}/yr
                                  {c.reward_type !== 'cashback' && ` (~${c.projected_points.toLocaleString()} ${c.reward_type})`}
                                </span>
                              </div>
                              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                <div className="h-full bg-emerald-400 rounded-full" style={{ width:`${pct}%` }}/>
                              </div>
                            </div>
                          );
                        })}
                        {rewardsAnalysis && rewardsAnalysis.every(c=>c.optimal_value===0) && (
                          <p className="text-xs text-slate-400">No projected rewards yet — add cards or upload a statement.</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Chat */}
                  <div className="lg:col-span-7 bg-white border border-slate-200 p-8 rounded-[32px] shadow-sm flex flex-col h-[620px]">
                    <h3 className="text-slate-800 font-bold mb-4 flex items-center gap-2 text-xl"><Bot className="text-blue-600"/> NeuroCard AI Advisor</h3>
                    <div className="flex-1 overflow-y-auto pr-2 mb-4 space-y-3">
                      {chatHistory.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role==='user'?'justify-end':'justify-start'}`}>
                          <div className={`max-w-[80%] p-4 rounded-2xl text-sm leading-relaxed ${msg.role==='user'?'bg-blue-600 text-white rounded-tr-sm':'bg-slate-100 text-slate-800 rounded-tl-sm'}`}>
                            {msg.content}
                          </div>
                        </div>
                      ))}
                      {isChatting && (
                        <div className="flex justify-start">
                          <div className="bg-slate-100 p-4 rounded-2xl rounded-tl-sm flex gap-2 items-center">
                            {[0,100,200].map(d=><div key={d} className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay:`${d}ms` }}/>)}
                          </div>
                        </div>
                      )}
                      <div ref={chatEndRef}/>
                    </div>
                    <form onSubmit={handleChatSubmit} className="relative flex items-center">
                      <input type="text" value={chatInput} onChange={e=>setChatInput(e.target.value)} disabled={isChatting}
                        placeholder="Ask about redeeming points, best cards, strategy..."
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl pl-5 pr-14 py-4 text-sm font-bold text-slate-800 outline-none focus:border-blue-500"/>
                      <button type="submit" disabled={isChatting} className="absolute right-2 bg-blue-600 text-white p-2.5 rounded-xl hover:bg-blue-700 disabled:opacity-50">
                        <Send size={18}/>
                      </button>
                    </form>
                  </div>
                </div>
              </div>
            )}

            {/* ── WALLET & SMART SWIPE ── */}
            {activeTab==='wallet' && (
              <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div className="lg:col-span-5 bg-white border border-slate-200 p-8 rounded-[32px] shadow-sm h-fit">
                  <h3 className="text-slate-800 font-bold mb-6 flex items-center gap-2 text-xl"><Wallet className="text-blue-600"/> Virtual Wallet</h3>
                  <div className="flex gap-2 mb-6">
                    <select value={newCardSelect} onChange={e=>setNewCardSelect(e.target.value)} className="flex-1 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2 text-sm font-bold text-slate-700 outline-none truncate">
                      {uniqueBanks.filter(b=>b!=='All').map(bank=>(
                        <optgroup key={bank} label={bank}>
                          {allCardsList.filter(c=>c.bank_name===bank).map(c=><option key={c.card_name} value={c.card_name}>{c.card_name}</option>)}
                        </optgroup>
                      ))}
                    </select>
                    <button onClick={addToWallet} className="bg-blue-600 text-white px-4 py-2 rounded-xl flex items-center gap-1 font-bold hover:bg-blue-700"><Plus size={18}/> Add</button>
                  </div>
                  <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
                    {data.owned_cards.length===0 && <p className="text-slate-400 text-sm font-bold text-center py-10">Wallet empty. Add cards above.</p>}
                    {data.owned_cards.map((cName, i) => {
                      const co = allCardsList.find(c=>c.card_name===cName);
                      return (
                        <div key={i} className="border border-slate-100 p-4 rounded-2xl flex justify-between items-center bg-slate-50">
                          <div className="flex items-center gap-3">
                            {co ? <BankLogo bankName={co.bank_name}/> : <CreditCard size={16} className="text-slate-400"/>}
                            <p className="font-bold text-slate-800 text-sm">{cName}</p>
                          </div>
                          <button onClick={()=>removeFromWallet(cName)} className="text-red-400 hover:text-red-600 p-2"><Trash2 size={16}/></button>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="lg:col-span-7 bg-slate-900 p-8 rounded-[32px] shadow-xl text-white h-fit">
                  <h3 className="font-bold mb-2 flex items-center gap-2 text-xl"><Zap className="text-amber-400 fill-amber-400"/> Smart-Swipe Engine</h3>
                  <p className="text-sm text-slate-400 mb-6">Tap a merchant or search any brand. AI picks the best card from your <strong className="text-white">{data.owned_cards.length}</strong> cards.</p>

                  <div className="flex gap-2 mb-5">
                    <div className="relative flex-1">
                      <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"/>
                      <input type="text" value={swipeSearch} onChange={e=>setSwipeSearch(e.target.value)}
                        onKeyDown={e=>e.key==='Enter'&&runSwipe()}
                        placeholder="Search any brand (Rapido, BigBazaar...)"
                        className="w-full bg-white/10 border border-white/20 rounded-xl pl-9 pr-4 py-3 text-sm font-bold text-white placeholder-slate-400 outline-none focus:border-blue-400"/>
                    </div>
                    <button onClick={()=>runSwipe()} disabled={swipeLoading||!swipeSearch.trim()}
                      className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white px-5 py-3 rounded-xl font-bold flex items-center gap-2">
                      {swipeLoading ? <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"/> : <Zap size={16}/>}
                      Go
                    </button>
                  </div>

                  <div className="flex gap-2 flex-wrap max-h-[200px] overflow-y-auto pr-1 mb-5">
                    {SWIPE_MERCHANTS.map(m=>(
                      <button key={m} onClick={()=>{setSwipeSearch(m);runSwipe(m);}} disabled={swipeLoading}
                        className="px-3 py-2 bg-white/10 border border-white/10 rounded-xl text-xs font-bold text-slate-300 hover:bg-blue-600 hover:text-white transition-all active:scale-95 flex items-center gap-1.5 disabled:opacity-40">
                        <Store size={12} className="opacity-60"/> {m}
                      </button>
                    ))}
                  </div>

                  {swipeLoading && (
                    <div className="p-5 bg-white/5 rounded-2xl flex items-center gap-3 animate-pulse">
                      <Brain size={20} className="text-blue-400"/>
                      <p className="text-sm text-slate-300 font-bold">NeuroCard AI is classifying the merchant...</p>
                    </div>
                  )}

                  {swipeResult && !swipeResult.error && !swipeLoading && (
                    <div className="p-6 bg-blue-600/20 border border-blue-500/30 rounded-3xl space-y-4 animate-in fade-in zoom-in duration-300">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="text-blue-400 text-[10px] font-bold uppercase mb-1">Best Card for {swipeResult.merchant}</p>
                          <p className="text-2xl font-black">{swipeResult.best_card}</p>
                          <p className="text-[10px] text-slate-400 font-bold uppercase mt-1">{swipeResult.bank}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-blue-400 text-[10px] font-bold uppercase mb-1">Yield on ₹10k</p>
                          <p className="text-3xl font-black text-emerald-400">₹{swipeResult.savings_on_10k}</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="bg-white/10 rounded-xl p-3 text-center">
                          <p className="text-[8px] uppercase font-bold text-slate-400 mb-1">Category</p>
                          <p className="text-sm font-black">{CAT_EMOJIS[swipeResult.category]} {swipeResult.category}</p>
                        </div>
                        <div className="bg-white/10 rounded-xl p-3 text-center">
                          <p className="text-[8px] uppercase font-bold text-slate-400 mb-1">Confidence</p>
                          <p className="text-sm font-black">{Math.round(swipeResult.confidence*100)}%</p>
                        </div>
                        <div className="bg-white/10 rounded-xl p-3 text-center">
                          <p className="text-[8px] uppercase font-bold text-slate-400 mb-1">Classified by</p>
                          <div className={`text-[9px] font-black uppercase px-2 py-1 rounded-lg text-white inline-block ${SOURCE_BADGE[swipeResult.source]?.color||'bg-slate-500'}`}>
                            {SOURCE_BADGE[swipeResult.source]?.label||swipeResult.source}
                          </div>
                        </div>
                      </div>
                      {swipeResult.reason && <p className="text-xs text-slate-300 font-medium">💡 {swipeResult.reason}</p>}
                    </div>
                  )}
                  {swipeResult?.error && !swipeLoading && (
                    <div className="p-4 bg-red-500/20 border border-red-500/30 rounded-2xl text-red-400 font-bold text-sm text-center">{swipeResult.error}</div>
                  )}
                </div>
              </div>
            )}

            {/* ── STRATEGY LAB ── */}
            {activeTab==='strategy' && (
              <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div className="lg:col-span-4 bg-white border border-slate-200 p-8 rounded-[32px] shadow-sm h-fit">
                  <h3 className="text-slate-800 font-bold mb-6 flex items-center gap-2"><TrendingUp size={18} className="text-blue-600"/> Scenario Simulator</h3>
                  {CATEGORY_ORDER.map(cat => {
                    const col = CAT_COLORS[cat]||CAT_COLORS.others;
                    return (
                      <div key={cat} className="mb-5">
                        <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase mb-2">
                          <span>{CAT_EMOJIS[cat]} {cat}</span>
                          <span className="text-slate-700">₹{sliders[cat]?.toLocaleString()||0}</span>
                        </div>
                        <input type="range" min="0" max="400000" step="5000"
                          className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-blue-600"
                          value={sliders[cat]||0}
                          onChange={e=>handleSliderChange(cat, e.target.value)}
                          onMouseUp={runManualAI}/>
                      </div>
                    );
                  })}
                </div>

                <div className="lg:col-span-8">
                  <div className="flex items-center justify-between mb-4 px-2">
                    <h3 className="text-xl font-bold text-slate-800">Simulated Recommendations</h3>
                    <div className="flex items-center gap-3">
                      {compareCards.length > 0 && (
                        <>
                          <span className="text-xs text-slate-500 font-bold">{compareCards.length} selected</span>
                          <button onClick={()=>compareCards.length>=2?setShowCompare(true):alert("Select at least 2 cards to compare.")}
                            disabled={compareCards.length<2}
                            className="flex items-center gap-2 bg-violet-600 hover:bg-violet-700 disabled:opacity-40 text-white text-xs font-bold px-4 py-2.5 rounded-xl">
                            <GitCompare size={14}/> Compare {compareCards.length} Cards
                          </button>
                          <button onClick={()=>setCompareCards([])} className="text-slate-400 hover:text-red-500 text-xs font-bold">Clear</button>
                        </>
                      )}
                    </div>
                  </div>
                  <p className="text-xs text-slate-400 font-bold mb-4 px-2">Click Compare on any card to add it — no cap on how many you can compare.</p>
                  {renderCardGrid(manualRecs, true)}
                </div>
              </div>
            )}
          </>
        )
      )}
    </div>
  );
};

export default App;
