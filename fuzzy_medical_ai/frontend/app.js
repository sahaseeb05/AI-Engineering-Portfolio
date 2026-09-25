const form = document.querySelector('#patient-form');
const scoreNode = document.querySelector('#risk-score');
const categoryNode = document.querySelector('#risk-category');
const summaryNode = document.querySelector('#score-summary');
const progressNode = document.querySelector('#gauge-progress');
const saveButton = document.querySelector('#save-record-button');
const saveBanner = document.querySelector('#save-banner');
const circumference = 2 * Math.PI * 84;
let currentAssessment = null;
progressNode.style.strokeDasharray = `${circumference}`;
progressNode.style.strokeDashoffset = `${circumference}`;

const labels = {
  age: 'Age', blood_pressure: 'Blood pressure', cholesterol: 'Cholesterol',
  blood_sugar: 'Blood sugar', heart_rate: 'Heart rate'
};

function updateClock() {
  document.querySelector('#clock').textContent = new Date().toISOString().slice(11, 19);
}
setInterval(updateClock, 1000);
updateClock();

function renderMemberships(memberships) {
  document.querySelector('#membership-list').innerHTML = Object.entries(memberships).map(([key, values]) => `
    <div class="membership-row"><div class="membership-label"><span>${labels[key]}</span><span class="mono">${Math.max(...Object.values(values)).toFixed(2)}</span></div>
    <div class="membership-bars">${Object.entries(values).map(([state, value]) => `<div class="membership-track"><span class="membership-state">${state}</span><span class="membership-bar"><i class="bar-${state.toLowerCase()}" style="width:${value * 100}%"></i></span><span class="mono">${value.toFixed(2)}</span></div>`).join('')}</div></div>`).join('');
}

function renderFactors(factors) {
  document.querySelector('#factors-body').innerHTML = factors.map((factor) => `
    <tr><td><strong>${factor.label}</strong><span class="table-key">${factor.feature}</span></td><td class="mono">${factor.value} <span class="unit">${factor.unit}</span></td><td><span class="state state-${factor.state.toLowerCase()}">${factor.state}</span></td><td><span class="strength"><i style="width:${factor.membership * 100}%"></i></span><span class="mono">${factor.membership.toFixed(2)}</span></td><td class="mono contribution">${(factor.contribution * 100).toFixed(1)}%</td></tr>`).join('');
}

async function assess(event) {
  event.preventDefault();
  const button = form.querySelector('button');
  button.disabled = true;
  button.textContent = 'Calculating';
  const data = Object.fromEntries(new FormData(form).entries());
  Object.keys(data).forEach((key) => { data[key] = Number(data[key]); });
  try {
    const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Assessment unavailable');
    const score = Number(result.overall_risk);
    scoreNode.textContent = Math.round(score);
    categoryNode.textContent = `${result.category} risk`;
    categoryNode.className = `risk-badge risk-${result.category.toLowerCase()}`;
    summaryNode.textContent = `${result.category} signal detected across five monitored inputs. Review the ranked factors below before clinical action.`;
    progressNode.style.strokeDashoffset = `${circumference * (1 - score / 100)}`;
    document.querySelector('#model-score').textContent = `${result.model_risk}%`;
    document.querySelector('#fuzzy-score').textContent = `${result.fuzzy_risk}%`;
    renderMemberships(result.fuzzy_memberships);
    renderFactors(result.key_contributing_factors);
    currentAssessment = { telemetry: data, risk_score: score };
    saveButton.disabled = false;
    saveBanner.classList.remove('visible');
    document.querySelector('#validation-copy').textContent = `Assessment updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    summaryNode.textContent = error.message;
    document.querySelector('#validation-copy').textContent = 'Service connection required';
  } finally {
    button.disabled = false;
    button.innerHTML = 'Run assessment <span>↗</span>';
  }
}

form.addEventListener('submit', assess);

saveButton.addEventListener('click', async () => {
  if (!currentAssessment) return;
  saveButton.disabled = true;
  saveButton.textContent = 'Saving';
  try {
    const response = await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...currentAssessment.telemetry, risk_score: currentAssessment.risk_score })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Record could not be saved');
    saveBanner.classList.add('visible');
    document.querySelector('#validation-copy').textContent = `Verified record ${result.feedback_id} stored`;
  } catch (error) {
    document.querySelector('#validation-copy').textContent = error.message;
  } finally {
    saveButton.disabled = false;
    saveButton.innerHTML = 'Save &amp; Verify Record <span>+</span>';
  }
});
