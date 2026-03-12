import { useState } from "react";
import { Link } from "wouter";
import { useListPatients, useCreatePatient, CreatePatientInputGender } from "@workspace/api-client-react";
import { useQueryClient } from "@tanstack/react-query";
import { Plus, Search, Eye, Calendar, Users } from "lucide-react";
import { format } from "date-fns";
import { Modal } from "@/components/ui-custom/Modal";

export default function Patients() {
  const queryClient = useQueryClient();
  const { data: patients, isLoading } = useListPatients();
  const createPatient = useCreatePatient({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['/api/patients'] });
        setIsCreateModalOpen(false);
        setNewPatient({ name: "", age: 30, gender: "other" });
      }
    }
  });

  const [search, setSearch] = useState("");
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newPatient, setNewPatient] = useState({ name: "", age: 30, gender: "other" as CreatePatientInputGender });

  const filteredPatients = patients?.filter(p => p.name.toLowerCase().includes(search.toLowerCase())) || [];

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    createPatient.mutate({ data: newPatient });
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-display font-bold text-slate-900">Patients</h1>
          <p className="text-slate-500 mt-1">Manage patient profiles and diagnostic history.</p>
        </div>
        <button 
          onClick={() => setIsCreateModalOpen(true)}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold bg-gradient-to-r from-primary to-blue-600 text-white shadow-md shadow-primary/20 hover:shadow-lg hover:shadow-primary/30 hover:-translate-y-0.5 transition-all"
        >
          <Plus className="h-5 w-5" />
          Add Patient
        </button>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex items-center">
          <div className="relative w-full max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
            <input 
              type="text"
              placeholder="Search patients by name..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-100">
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Patient Name</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Age/Gender</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Added Date</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {isLoading ? (
                <tr>
                  <td colSpan={4} className="px-6 py-8 text-center text-slate-500">Loading patients...</td>
                </tr>
              ) : filteredPatients.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-6 py-12 text-center">
                    <Users className="h-12 w-12 text-slate-300 mx-auto mb-3" />
                    <p className="text-slate-500 font-medium">No patients found.</p>
                  </td>
                </tr>
              ) : (
                filteredPatients.map(patient => (
                  <tr key={patient.id} className="hover:bg-slate-50/80 transition-colors group">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-primary/10 text-primary flex items-center justify-center font-bold">
                          {patient.name.charAt(0)}
                        </div>
                        <span className="font-semibold text-slate-900 group-hover:text-primary transition-colors">{patient.name}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-slate-600">
                      {patient.age} yrs • <span className="capitalize">{patient.gender}</span>
                    </td>
                    <td className="px-6 py-4 text-slate-500 flex items-center gap-2">
                      <Calendar className="h-4 w-4" />
                      {format(new Date(patient.createdAt), 'MMM dd, yyyy')}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <Link href={`/patients/${patient.id}`}>
                        <button className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-primary hover:bg-primary/10 rounded-lg transition-colors">
                          <Eye className="h-4 w-4" />
                          View
                        </button>
                      </Link>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <Modal isOpen={isCreateModalOpen} onClose={() => setIsCreateModalOpen(false)} title="Add New Patient">
        <form onSubmit={handleCreate} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Full Name</label>
            <input 
              required
              type="text"
              value={newPatient.name}
              onChange={e => setNewPatient({...newPatient, name: e.target.value})}
              className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all"
              placeholder="e.g. John Doe"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Age</label>
              <input 
                required
                type="number"
                min="0"
                max="120"
                value={newPatient.age}
                onChange={e => setNewPatient({...newPatient, age: parseInt(e.target.value)})}
                className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Gender</label>
              <select 
                value={newPatient.gender}
                onChange={e => setNewPatient({...newPatient, gender: e.target.value as CreatePatientInputGender})}
                className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all"
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
          <div className="pt-4 flex justify-end gap-3">
            <button 
              type="button" 
              onClick={() => setIsCreateModalOpen(false)}
              className="px-4 py-2.5 rounded-xl font-medium text-slate-600 hover:bg-slate-100 transition-colors"
            >
              Cancel
            </button>
            <button 
              type="submit"
              disabled={createPatient.isPending}
              className="px-6 py-2.5 rounded-xl font-semibold bg-primary text-white hover:bg-primary/90 disabled:opacity-50 transition-colors"
            >
              {createPatient.isPending ? "Creating..." : "Save Patient"}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
